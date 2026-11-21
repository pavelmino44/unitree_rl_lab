"""Go1 ground properties authored at scene construction, before PhysX parsing."""
from __future__ import annotations

import json

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaacsim.core.utils.stage import get_current_stage
from pxr import PhysxSchema, Usd, UsdPhysics


class Go1TerrainImporter(TerrainImporter):
    def __init__(self, cfg):
        sim = sim_utils.SimulationContext.instance()
        if sim is None or sim.is_playing():
            raise RuntimeError("Construct Go1 terrain before starting the simulation")
        super().__init__(cfg)
        # The floor is not parsed by PhysX yet. Both final values will be read
        # together, without invalid intermediate live updates of the pair.
        sim_utils.modify_collision_properties(
            cfg.prim_path,
            sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
        )
        stage = get_current_stage()
        offsets = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(cfg.prim_path)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision = PhysxSchema.PhysxCollisionAPI(prim)
            contact = collision.GetContactOffsetAttr().Get()
            rest = collision.GetRestOffsetAttr().Get()
            if contact is None or rest is None or abs(contact-0.0005) > 1e-8 or abs(rest) > 1e-9:
                raise RuntimeError(f"Invalid Go1 ground offsets at {prim.GetPath()}: contact={contact}, rest={rest}")
            offsets.append({"prim": str(prim.GetPath()), "contact_offset_m": contact, "rest_offset_m": rest})
        if not offsets:
            raise RuntimeError(f"No Go1 ground colliders found under {cfg.prim_path}")
        self.go1_ground_offsets = offsets
        print("[GO1 GROUND] authored before PhysX start: " + json.dumps(offsets), flush=True)
