# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cosmos H2 world model whose frames also carry the fingertip tactile maps.

The world model is trained on a composite frame: the head camera on top and the
ten fingertip deformation maps as tiles below it. The layout, and the resizing
between tiles and maps, come from the Cosmos tree the model was trained with,
so the environment composes and reads frames exactly as training did.
"""

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.datasets.lerobot_world_model import LeRobotV21InitDataset
from rlinf.envs.sim.world_model.cosmos3_env import Cosmos3Env


class LeRobotV21TilesInitDataset(LeRobotV21InitDataset):
    """Reset frames composed of the head camera and the tactile map tiles.

    Args:
        *args: Positional arguments of :class:`LeRobotV21InitDataset`.
        tile: Side of one tactile tile in pixels.
        gap: Black rows between the head camera and the tiles.
        image_size: Height and width of the composite frame.
        **kwargs: Remaining arguments of :class:`LeRobotV21InitDataset`.
    """

    def __init__(
        self,
        *args,
        tile: int,
        gap: int,
        image_size: tuple[int, int],
        **kwargs,
    ):
        # The head frame keeps its own size; only the composite has image_size.
        super().__init__(*args, image_size=None, **kwargs)
        self.tile, self.gap = int(tile), int(gap)
        self.canvas_size = tuple(image_size)

    def _tactile_frame(self, episode: int, key: str, frame: int) -> np.ndarray:
        """Return one deformation map ``[H, W]`` uint8 of a tactile video."""
        import av

        path = self.root / self.info["video_path"].format(
            episode_chunk=episode // self.chunks_size,
            episode_index=episode,
            video_key=key,
        )
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            for index, decoded in enumerate(container.decode(stream)):
                if index == frame:
                    # The maps are gray; the video stores them in every channel.
                    return decoded.to_ndarray(format="rgb24")[..., 0]
        raise ValueError(f"Frame {frame} is missing from {path}")

    def get_at_frame(
        self, index: int, start_frame: int | None = None
    ) -> dict[str, Any]:
        """Load one reset whose image is the composite frame."""
        from cosmos_framework.utils.h2_tiles_layout import (
            TACTILE_KEYS,
            H2TilesLayout,
            compose,
        )

        item = super().get_at_frame(index, start_frame)
        layout = H2TilesLayout(tile=self.tile, gap=self.gap)
        if (layout.canvas_h, layout.canvas_w) != self.canvas_size:
            raise ValueError(
                f"Expected image_size {(layout.canvas_h, layout.canvas_w)} for tile "
                f"{self.tile} and gap {self.gap}, found {self.canvas_size}"
            )
        start = item["start_items"][0]
        frame = item["dataset_meta"]["start_frame"]
        head = (start["image"] * 255).round().byte().permute(1, 2, 0).numpy()
        maps = np.stack(
            [
                self._tactile_frame(item["episode_index"], key, frame)
                for key in TACTILE_KEYS
            ]
        )
        canvas = compose(head[None], maps[None], layout)[0]
        start["image"] = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255
        return item


class TactileForceReadout:
    """Fingertip forces read from deformation maps.

    A frozen Sharpa map encoder gives one feature per fingertip, and a linear
    map per fingertip, fitted on recorded data, turns it into a six-axis force.

    Args:
        encoder_checkpoint: Sharpa deformation-map encoder weights.
        weights: ``.npz`` with ``mu`` and ``sd`` ``[N, 512]``, ``Wf``
            ``[N, 512, 6]`` and ``bf`` ``[N, 6]``.
        device: Device the encoder runs on.
    """

    def __init__(self, encoder_checkpoint: str, weights: str, device: torch.device):
        from gr00t.model.modules.tactile.sharpa_encoder import SharpaTactileEncoder

        self.device = device
        self.encoder = SharpaTactileEncoder(encoder_checkpoint).to(device)
        arrays = np.load(Path(weights).expanduser())
        self.mean, self.std, self.weight, self.bias = (
            torch.from_numpy(arrays[name]).float().to(device)
            for name in ("mu", "sd", "Wf", "bf")
        )

    @torch.no_grad()
    def __call__(self, maps: torch.Tensor) -> torch.Tensor:
        """Map ``[M, N, 240, 240]`` uint8 maps to ``[M, 6N]`` forces."""
        forces = []
        for batch in maps.split(64):
            features = self.encoder(batch[:, :, None].to(self.device).float())
            scaled = (features - self.mean) / self.std
            force = torch.einsum("mnd,ndf->mnf", scaled, self.weight) + self.bias
            forces.append(force.flatten(1))
        return torch.cat(forces)


class Cosmos3TactileEnv(Cosmos3Env):
    """Cosmos H2 environment that also observes fingertip maps and forces.

    ``current_obs`` holds composite frames. The policy gets the head camera
    region as its image and, under ``tactile``, the ten deformation maps of the
    last frame and the force window read from the most recent frames.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        record_metrics=True,
        worker_info=None,
    ):
        tactile = cfg.tactile
        self.reset_dataset_cls = partial(
            LeRobotV21TilesInitDataset, tile=tactile.tile, gap=tactile.gap
        )
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            record_metrics=record_metrics,
            worker_info=worker_info,
        )
        from cosmos_framework.utils.h2_tiles_layout import (
            H2TilesLayout,
            register_canvas,
        )

        self.layout = H2TilesLayout(tile=tactile.tile, gap=tactile.gap)
        # The model snaps a frame to a known canvas; the composite must be one.
        register_canvas(self.layout, "480")
        self.force_window = int(tactile.force_window)
        if self.force_window > self.chunk:
            raise ValueError(
                f"Expected force_window <= chunk ({self.chunk}), found "
                f"{self.force_window}"
            )
        self.force_readout = TactileForceReadout(
            tactile.encoder_checkpoint, tactile.force_readout, self.device
        )

    def _canvas_frames(self, count: int) -> torch.Tensor:
        """Return the last ``count`` composite frames as uint8 ``[B, T, H, W, 3]``."""
        frames = self.current_obs[:, :, 0, -count:]
        return (
            ((frames + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 4, 1)
        )

    def _tactile_obs(self) -> dict[str, torch.Tensor]:
        """Read the maps of the last frame and the force window from the tiles."""
        from cosmos_framework.utils.h2_tiles_layout import crop_tiles, upsample_tiles

        canvas = self._canvas_frames(self.force_window)
        num_envs, num_frames = canvas.shape[:2]
        tiles = crop_tiles(canvas.flatten(0, 1).cpu().numpy(), self.layout)
        maps = torch.from_numpy(upsample_tiles(tiles))
        forces = self.force_readout(maps).view(num_envs, num_frames, -1)
        # Before the first chunk only the reset frame exists; it fills the window.
        if num_frames < self.force_window:
            pad = forces[:, :1].expand(-1, self.force_window - num_frames, -1)
            forces = torch.cat([pad, forces], dim=1)
        maps = maps.view(num_envs, num_frames, *maps.shape[1:])[:, -1]
        return {"deform_maps": maps.to(self.device), "forces": forces}

    def _wrap_obs(self):
        from cosmos_framework.utils.h2_tiles_layout import HEAD_H, HEAD_W

        obs = super()._wrap_obs()
        obs["main_images"] = obs["main_images"][:, :HEAD_H, :HEAD_W]
        obs["tactile"] = self._tactile_obs()
        return obs

    def capture_image(self):
        if self._render_frames is not None:
            return self._render_frames
        return self._canvas_frames(1).cpu()
