from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class ImportedObject:
    name: str
    category: str
    mesh: Path
    texture: Path | None
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    fixed: bool
    mass: float
    friction: tuple[float, float, float]


@dataclass(frozen=True)
class ImportedSimFoundryScene:
    scene_manifest: Path
    external_mjcf: Path
    semantic_map: Path
    objects: tuple[ImportedObject, ...]


class SimFoundryMuJoCoImporter:
    """Compile a SimFoundry saved-scene document into an exp-agent bundle.

    SimFoundry USD remains the source of truth. Meshes are exported into a
    generated cache; source USDs and scene JSON are never edited. Gaussian
    backgrounds are supplied independently because they are visual-only, while
    the generated MJCF contains physical ground, collision geometry and
    individually movable rigid objects.
    """

    def __init__(self, scene_state: str | Path, output_directory: str | Path) -> None:
        self.scene_state = Path(scene_state).expanduser().resolve()
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.raw: dict[str, Any] = json.loads(self.scene_state.read_text(encoding="utf-8"))

    def import_scene(
        self,
        *,
        gaussian_splat: str | Path | None = None,
        gaussian_alignment: list[list[float]] | None = None,
        collision_mesh: str | Path | None = None,
        navigation_grid: str | Path | None = None,
        include_background_mesh: bool = True,
        scene_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> ImportedSimFoundryScene:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        mesh_dir = self.output_directory / "meshes"
        mesh_dir.mkdir(exist_ok=True)
        grid = self._navigation_grid(navigation_grid)
        self._add_support_to_navigation_grid(grid, scene_offset)
        objects = tuple(self._objects(mesh_dir, include_background_mesh, scene_offset))
        external = self.output_directory / "simfoundry_scene.xml"
        self._write_mjcf(external, objects, collision_mesh, scene_offset, grid)
        semantics = self.output_directory / "semantic_objects.json"
        semantics.write_text(
            json.dumps(
                {
                    "locations": self._semantic_locations(objects, grid)
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = self.output_directory / "scene_bundle.json"
        payload: dict[str, Any] = {
            "external_mjcf": external.name,
            "navigation_grid": grid.name,
            "semantic_map": semantics.name,
            "navigation_footprint_radius_m": 0.2,
            "initial_pose": self._initial_pose(grid),
            "source": {
                "format": "simfoundry_saved_scene",
                "scene_state": str(self.scene_state),
            },
        }
        if gaussian_splat is not None:
            payload["gaussian_splat"] = str(Path(gaussian_splat).expanduser().resolve())
            payload["gaussian_alignment"] = gaussian_alignment or self._identity()
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return ImportedSimFoundryScene(manifest, external, semantics, objects)

    def _objects(
        self,
        mesh_dir: Path,
        include_background: bool,
        scene_offset: tuple[float, float, float],
    ) -> list[ImportedObject]:
        definitions = self.raw.get("objects_info", {}).get("init_info", {})
        states = self.raw.get("state", {}).get("registry", {}).get("object_registry", {})
        result: list[ImportedObject] = []
        for name, definition in definitions.items():
            args = definition.get("args", {})
            category = str(args.get("category", "object"))
            if name.startswith("robot") or definition.get("class_name") in {"Yam", "DROID"}:
                continue
            if category == "mesh_background" and not include_background:
                continue
            asset = args.get("usd_path")
            if not asset:
                continue
            source = self._resolve(asset)
            mesh = self._mesh(source, mesh_dir / f"{self._safe(name)}.obj")
            state = states.get(name, {}).get("root_link", {})
            raw_pos = self._tuple(state.get("pos", (0, 0, 0)), 3, "position")
            pos = tuple(raw_pos[index] + scene_offset[index] for index in range(3))
            xyzw = self._tuple(state.get("ori", (0, 0, 0, 1)), 4, "orientation")
            scale = self._scale(args.get("scale", (1, 1, 1)))
            fixed = bool(args.get("fixed_base", category == "mesh_background"))
            friction = self._friction(args)
            texture = self._texture(source)
            result.append(
                ImportedObject(
                    name=str(name),
                    category=category,
                    mesh=mesh,
                    texture=texture,
                    position=pos,
                    quaternion_wxyz=(xyzw[3], xyzw[0], xyzw[1], xyzw[2]),
                    scale=scale,
                    fixed=fixed,
                    mass=max(0.001, float(args.get("mass", 0.2))),
                    friction=friction,
                )
            )
        return result

    def _write_mjcf(
        self,
        path: Path,
        objects: tuple[ImportedObject, ...],
        collision_mesh: str | Path | None,
        scene_offset: tuple[float, float, float],
        navigation_grid: Path,
    ) -> None:
        root = ET.Element("mujoco", {"model": "imported SimFoundry scene"})
        ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": ""})
        assets = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", {"name": "key", "pos": "0 0 3", "dir": "0 0 -1"})
        ground = self.raw.get("ground_plane_info", {})
        ground_raw = self._tuple(ground.get("position", (0, 0, 0)), 3, "ground position")
        ground_pos = tuple(ground_raw[index] + scene_offset[index] for index in range(3))
        # The humanoid still needs a world floor even when the imported scene's
        # support plane is elevated to become a table surface.
        ET.SubElement(
            world,
            "geom",
            {"name": "robot_floor", "type": "plane", "pos": "0 0 0", "size": "10 10 0.05", "friction": "1 0.005 0.0001", "rgba": "0.18 0.18 0.18 1"},
        )
        ET.SubElement(
            world,
            "geom",
            {
                "name": "simfoundry_support",
                "type": "box",
                "pos": self._values(ground_pos),
                "size": "0.75 0.75 0.01",
                "friction": "1 0.005 0.0001",
                "rgba": "0.18 0.18 0.18 1",
            },
        )
        for item in objects:
            mesh_name = f"mesh_{self._safe(item.name)}"
            ET.SubElement(
                assets,
                "mesh",
                {"name": mesh_name, "file": str(item.mesh), "scale": self._values(item.scale)},
            )
            material = None
            if item.texture is not None:
                texture_name = f"texture_{self._safe(item.name)}"
                material = f"material_{self._safe(item.name)}"
                ET.SubElement(assets, "texture", {"name": texture_name, "type": "2d", "file": str(item.texture)})
                ET.SubElement(assets, "material", {"name": material, "texture": texture_name})
            geom_args = {
                "name": item.name,
                "type": "mesh",
                "mesh": mesh_name,
                "friction": self._values(item.friction),
            }
            if material:
                geom_args["material"] = material
            if item.fixed:
                geom_args.update(
                    {"pos": self._values(item.position), "quat": self._values(item.quaternion_wxyz)}
                )
                ET.SubElement(world, "geom", geom_args)
            else:
                body = ET.SubElement(
                    world,
                    "body",
                    {
                        "name": item.name,
                        "pos": self._values(item.position),
                        "quat": self._values(item.quaternion_wxyz),
                    },
                )
                ET.SubElement(body, "freejoint", {"name": f"{item.name}_free"})
                geom_args["mass"] = str(item.mass)
                ET.SubElement(body, "geom", geom_args)
        if collision_mesh is not None:
            source = Path(collision_mesh).expanduser().resolve()
            converted = self._mesh(source, self.output_directory / "meshes" / "collision.obj")
            ET.SubElement(assets, "mesh", {"name": "background_collision", "file": str(converted)})
            ET.SubElement(
                world,
                "geom",
                {
                    "name": "background_collision",
                    "type": "mesh",
                    "mesh": "background_collision",
                    "rgba": "0.3 0.3 0.3 0",
                    "friction": "1 0.005 0.0001",
                },
            )
        # The navigation representation is also a conservative physical proxy
        # when a source scene has no triangle collision mesh. Gaussian pixels
        # never become physics; occupied cells become invisible wall boxes.
        grid = json.loads(navigation_grid.read_text(encoding="utf-8"))
        resolution = float(grid["resolution"])
        origin = grid.get("origin", [0.0, 0.0])
        for index, cell in enumerate(grid.get("occupied", [])):
            x = float(origin[0]) + (float(cell[0]) + 0.5) * resolution
            y = float(origin[1]) + (float(cell[1]) + 0.5) * resolution
            ET.SubElement(
                world,
                "geom",
                {
                    "name": f"navigation_collision_{index}",
                    "type": "box",
                    "pos": self._values((x, y, 1.0)),
                    "size": self._values((resolution * 0.5, resolution * 0.5, 1.0)),
                    "rgba": "0 0 0 0",
                    "friction": "1 0.005 0.0001",
                },
            )
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="unicode")

    def _mesh(self, source: Path, destination: Path) -> Path:
        if source.suffix.lower() == ".obj":
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return destination
        if source.suffix.lower() == ".ply":
            return self._ply_to_obj(source, destination)
        if source.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
            return self._usd_to_obj(source, destination)
        raise ValueError(f"unsupported SimFoundry mesh format: {source}")

    @staticmethod
    def _usd_to_obj(source: Path, destination: Path) -> Path:
        try:
            from pxr import Usd, UsdGeom
        except ImportError as exc:
            raise ImportError(
                "USD conversion needs the standalone usd-core package; run the importer in "
                "exp-agent's scene-import environment"
            ) from exc
        stage = Usd.Stage.Open(str(source))
        if stage is None:
            raise ValueError(f"could not open USD: {source}")
        cache = UsdGeom.XformCache()
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            matrix = cache.GetLocalToWorldTransform(prim)
            offset = len(vertices) + 1
            vertices.extend(tuple(float(v) for v in matrix.Transform(point)) for point in points)
            indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
            cursor = 0
            for count in mesh.GetFaceVertexCountsAttr().Get() or []:
                polygon = indices[cursor : cursor + count]
                cursor += count
                for index in range(1, len(polygon) - 1):
                    faces.append((offset + polygon[0], offset + polygon[index], offset + polygon[index + 1]))
        if not vertices or not faces:
            raise ValueError(f"USD contains no triangle meshes: {source}")
        SimFoundryMuJoCoImporter._write_obj(destination, vertices, faces)
        return destination

    @staticmethod
    def _ply_to_obj(source: Path, destination: Path) -> Path:
        try:
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError("PLY conversion needs plyfile") from exc
        data = PlyData.read(str(source))
        vertex = data["vertex"]
        vertices = list(zip(vertex["x"], vertex["y"], vertex["z"]))
        faces: list[tuple[int, int, int]] = []
        for record in data["face"]:
            polygon = record[0]
            for index in range(1, len(polygon) - 1):
                faces.append((int(polygon[0]) + 1, int(polygon[index]) + 1, int(polygon[index + 1]) + 1))
        SimFoundryMuJoCoImporter._write_obj(destination, vertices, faces)
        return destination

    @staticmethod
    def _write_obj(path: Path, vertices: Any, faces: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii") as handle:
            handle.writelines(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in vertices)
            handle.writelines(f"f {a} {b} {c}\n" for a, b, c in faces)

    def _navigation_grid(self, supplied: str | Path | None) -> Path:
        destination = self.output_directory / "navigation_grid.json"
        if supplied is not None:
            shutil.copy2(Path(supplied).expanduser().resolve(), destination)
        else:
            destination.write_text(
                json.dumps({"width": 40, "height": 40, "resolution": 0.1, "origin": [-2, -2], "occupied": []}, indent=2) + "\n",
                encoding="utf-8",
            )
        return destination

    def _add_support_to_navigation_grid(
        self, grid_path: Path, scene_offset: tuple[float, float, float]
    ) -> None:
        """Project the imported support into the planner's collision layer."""
        raw = json.loads(grid_path.read_text(encoding="utf-8"))
        resolution = float(raw["resolution"])
        origin_x, origin_y = (float(value) for value in raw.get("origin", [0.0, 0.0]))
        ground = self.raw.get("ground_plane_info", {})
        ground_raw = self._tuple(ground.get("position", (0, 0, 0)), 3, "ground position")
        center_x = ground_raw[0] + scene_offset[0]
        center_y = ground_raw[1] + scene_offset[1]
        occupied = {(int(cell[0]), int(cell[1])) for cell in raw.get("occupied", [])}
        for x in range(int(raw["width"])):
            world_x = origin_x + (x + 0.5) * resolution
            if abs(world_x - center_x) > 0.75:
                continue
            for y in range(int(raw["height"])):
                world_y = origin_y + (y + 0.5) * resolution
                if abs(world_y - center_y) <= 0.75:
                    occupied.add((x, y))
        raw["occupied"] = [list(cell) for cell in sorted(occupied, key=lambda cell: (cell[1], cell[0]))]
        raw.setdefault("provenance", {})["simfoundry_support_projection"] = {
            "center": [center_x, center_y],
            "half_extents": [0.75, 0.75],
        }
        grid_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _semantic_locations(
        objects: tuple[ImportedObject, ...], grid_path: Path
    ) -> dict[str, dict[str, Any]]:
        raw = json.loads(grid_path.read_text(encoding="utf-8"))
        width, height = int(raw["width"]), int(raw["height"])
        resolution = float(raw["resolution"])
        origin_x, origin_y = (float(value) for value in raw.get("origin", [0.0, 0.0]))
        occupied = {(int(cell[0]), int(cell[1])) for cell in raw.get("occupied", [])}
        radius = int(math.ceil(0.2 / resolution))
        inflated = set(occupied)
        for x, y in occupied:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        candidate = (x + dx, y + dy)
                        if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                            inflated.add(candidate)

        locations: dict[str, dict[str, Any]] = {}
        free = [
            (origin_x + (x + 0.5) * resolution, origin_y + (y + 0.5) * resolution)
            for x in range(width)
            for y in range(height)
            if (x, y) not in inflated
        ]
        for item in objects:
            locations[item.name] = {
                "x": item.position[0], "y": item.position[1], "z": item.position[2],
                "yaw": SimFoundryMuJoCoImporter._yaw(item.quaternion_wxyz),
                "category": item.category, "interactive": not item.fixed,
                "navigation_target": False,
            }
            if item.fixed or not free:
                continue
            approach_x, approach_y = min(
                free,
                key=lambda point: math.hypot(point[0] - item.position[0], point[1] - item.position[1]),
            )
            locations[f"{item.name}_approach"] = {
                "x": approach_x,
                "y": approach_y,
                "z": 0.0,
                "yaw": math.atan2(item.position[1] - approach_y, item.position[0] - approach_x),
                "category": f"{item.category}_approach",
                "interactive": False,
                "target_object": item.name,
                "navigation_target": True,
            }
        return locations

    @staticmethod
    def _initial_pose(grid_path: Path) -> dict[str, Any]:
        """Choose the nearest origin-relative cell with humanoid stop clearance."""
        raw = json.loads(grid_path.read_text(encoding="utf-8"))
        width, height = int(raw["width"]), int(raw["height"])
        resolution = float(raw["resolution"])
        origin_x, origin_y = (float(value) for value in raw.get("origin", [0.0, 0.0]))
        occupied = {(int(cell[0]), int(cell[1])) for cell in raw.get("occupied", [])}
        required_cells = max(1, math.ceil(0.35 / resolution))

        def safe(cell: tuple[int, int]) -> bool:
            x, y = cell
            if min(x, y, width - 1 - x, height - 1 - y) < required_cells:
                return False
            return all(
                (x + dx, y + dy) not in occupied
                for dx in range(-required_cells, required_cells + 1)
                for dy in range(-required_cells, required_cells + 1)
                if dx * dx + dy * dy <= required_cells * required_cells
            )

        candidates = [
            (x, y) for x in range(width) for y in range(height) if safe((x, y))
        ]
        if not candidates:
            raise ValueError("SimFoundry navigation grid has no collision-safe initial pose")
        cell = min(
            candidates,
            key=lambda value: math.hypot(
                origin_x + (value[0] + 0.5) * resolution,
                origin_y + (value[1] + 0.5) * resolution,
            ),
        )
        return {
            "x": origin_x + (cell[0] + 0.5) * resolution,
            "y": origin_y + (cell[1] + 0.5) * resolution,
            "yaw": 0.0,
            "selection": "nearest_origin_collision_safe_cell",
        }

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        path = path if path.is_absolute() else self.scene_state.parent / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"SimFoundry asset does not exist: {path}")
        return path

    @staticmethod
    def _texture(source: Path) -> Path | None:
        for parent in source.parents:
            material = parent / "material" / "material_0.png"
            if material.exists():
                return material
            if parent.name == "objects":
                break
        return None

    @staticmethod
    def _friction(args: dict[str, Any]) -> tuple[float, float, float]:
        materials = args.get("link_physics_materials", {})
        first = next(iter(materials.values()), {}) if isinstance(materials, dict) else {}
        sliding = float(first.get("dynamic_friction", first.get("static_friction", 0.8)))
        return (sliding, 0.005, 0.0001)

    @staticmethod
    def _scale(value: Any) -> tuple[float, float, float]:
        if isinstance(value, (int, float)):
            return (float(value),) * 3
        return SimFoundryMuJoCoImporter._tuple(value, 3, "scale")

    @staticmethod
    def _tuple(value: Any, length: int, label: str) -> tuple[Any, ...]:
        result = tuple(float(item) for item in value)
        if len(result) != length:
            raise ValueError(f"{label} must contain {length} values")
        return result

    @staticmethod
    def _values(values: tuple[Any, ...]) -> str:
        return " ".join(f"{float(value):.9g}" for value in values)

    @staticmethod
    def _safe(value: str) -> str:
        return "".join(character if character.isalnum() or character == "_" else "_" for character in value)

    @staticmethod
    def _identity() -> list[list[float]]:
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    @staticmethod
    def _yaw(quaternion: tuple[float, float, float, float]) -> float:
        w, x, y, z = quaternion
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a SimFoundry saved scene into MuJoCo")
    parser.add_argument("scene_state", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--gaussian-splat", type=Path)
    parser.add_argument("--collision-mesh", type=Path)
    parser.add_argument("--navigation-grid", type=Path)
    parser.add_argument("--without-background-mesh", action="store_true")
    parser.add_argument("--scene-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--gaussian-alignment-json",
        type=Path,
        help="JSON containing a 4x4 matrix or a scene manifest with gaussian_alignment",
    )
    args = parser.parse_args()
    alignment = None
    if args.gaussian_alignment_json is not None:
        alignment_raw = json.loads(args.gaussian_alignment_json.read_text(encoding="utf-8"))
        alignment = alignment_raw.get("gaussian_alignment", alignment_raw)
    result = SimFoundryMuJoCoImporter(args.scene_state, args.output_directory).import_scene(
        gaussian_splat=args.gaussian_splat,
        gaussian_alignment=alignment,
        collision_mesh=args.collision_mesh,
        navigation_grid=args.navigation_grid,
        include_background_mesh=not args.without_background_mesh,
        scene_offset=tuple(args.scene_offset),
    )
    print(json.dumps({"scene_manifest": str(result.scene_manifest), "objects": [item.name for item in result.objects]}, indent=2))


if __name__ == "__main__":
    main()
