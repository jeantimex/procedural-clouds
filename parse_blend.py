"""Extract shader node tree details from the blend file."""
import bpy
import json
import sys

results = []

for mat in bpy.data.materials:
    if not mat.use_nodes or not mat.node_tree:
        continue

    mat_info = {
        "material": mat.name,
        "nodes": [],
        "links": [],
    }

    for node in mat.node_tree.nodes:
        node_info = {
            "name": node.name,
            "type": node.type,
            "bl_idname": node.bl_idname,
            "label": node.label,
            "location": [node.location.x, node.location.y],
            "inputs": {},
            "outputs": {},
        }

        # Extract input values
        for inp in node.inputs:
            val = None
            if hasattr(inp, "default_value"):
                dv = inp.default_value
                try:
                    val = list(dv)
                except TypeError:
                    val = dv
            node_info["inputs"][inp.name] = {
                "type": inp.type,
                "default_value": val,
            }

        # Extract output info
        for out in node.outputs:
            node_info["outputs"][out.name] = {"type": out.type}

        # Node-specific properties
        if node.type == "TEX_NOISE":
            node_info["noise_dimensions"] = node.noise_dimensions
            node_info["noise_type"] = node.noise_type if hasattr(node, "noise_type") else None
        elif node.type == "TEX_VORONOI":
            node_info["voronoi_dimensions"] = node.voronoi_dimensions
            node_info["distance"] = node.distance
            node_info["feature"] = node.feature
        elif node.type == "MAPPING":
            pass  # inputs carry the values
        elif node.type == "MATH":
            node_info["operation"] = node.operation
            node_info["use_clamp"] = node.use_clamp
        elif node.type == "MIX_RGB" or node.bl_idname == "ShaderNodeMix":
            if hasattr(node, "blend_type"):
                node_info["blend_type"] = node.blend_type
            if hasattr(node, "data_type"):
                node_info["data_type"] = node.data_type
            if hasattr(node, "clamp_factor"):
                node_info["clamp_factor"] = node.clamp_factor
            if hasattr(node, "clamp_result"):
                node_info["clamp_result"] = node.clamp_result
        elif node.type == "VECT_MATH":
            node_info["operation"] = node.operation
        elif node.type == "MAP_RANGE":
            if hasattr(node, "data_type"):
                node_info["data_type"] = node.data_type
            if hasattr(node, "interpolation_type"):
                node_info["interpolation_type"] = node.interpolation_type
            node_info["clamp"] = node.clamp
        elif node.type == "COLORRAMP" or node.type == "VALTORGB":
            cr = node.color_ramp
            node_info["color_ramp"] = {
                "interpolation": cr.interpolation,
                "color_mode": cr.color_mode,
                "elements": [
                    {"position": e.position, "color": list(e.color)}
                    for e in cr.elements
                ],
            }
        elif node.type == "TEX_COORD":
            pass
        elif node.type == "COMBXYZ" or node.type == "SEPXYZ":
            pass
        elif node.type in ("BSDF_PRINCIPLED", "OUTPUT_MATERIAL"):
            pass
        elif node.type == "GROUP":
            node_info["node_tree_name"] = node.node_tree.name if node.node_tree else None

        mat_info["nodes"].append(node_info)

    # Extract links
    for link in mat.node_tree.links:
        mat_info["links"].append({
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.name,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.name,
        })

    results.append(mat_info)

# Also extract any node groups
groups = []
for ng in bpy.data.node_groups:
    group_info = {
        "name": ng.name,
        "nodes": [],
        "links": [],
        "group_inputs": [],
        "group_outputs": [],
    }

    for node in ng.nodes:
        node_info = {
            "name": node.name,
            "type": node.type,
            "bl_idname": node.bl_idname,
            "label": node.label,
            "location": [node.location.x, node.location.y],
            "inputs": {},
            "outputs": {},
        }

        for inp in node.inputs:
            val = None
            if hasattr(inp, "default_value"):
                dv = inp.default_value
                try:
                    val = list(dv)
                except TypeError:
                    val = dv
            node_info["inputs"][inp.name] = {
                "type": inp.type,
                "default_value": val,
            }

        for out in node.outputs:
            node_info["outputs"][out.name] = {"type": out.type}

        # Same property extraction as above
        if node.type == "TEX_NOISE":
            node_info["noise_dimensions"] = node.noise_dimensions
            node_info["noise_type"] = node.noise_type if hasattr(node, "noise_type") else None
        elif node.type == "TEX_VORONOI":
            node_info["voronoi_dimensions"] = node.voronoi_dimensions
            node_info["distance"] = node.distance
            node_info["feature"] = node.feature
        elif node.type == "MATH":
            node_info["operation"] = node.operation
            node_info["use_clamp"] = node.use_clamp
        elif node.type == "MIX_RGB" or node.bl_idname == "ShaderNodeMix":
            if hasattr(node, "blend_type"):
                node_info["blend_type"] = node.blend_type
            if hasattr(node, "data_type"):
                node_info["data_type"] = node.data_type
            if hasattr(node, "clamp_factor"):
                node_info["clamp_factor"] = node.clamp_factor
            if hasattr(node, "clamp_result"):
                node_info["clamp_result"] = node.clamp_result
        elif node.type == "VECT_MATH":
            node_info["operation"] = node.operation
        elif node.type == "MAP_RANGE":
            if hasattr(node, "data_type"):
                node_info["data_type"] = node.data_type
            if hasattr(node, "interpolation_type"):
                node_info["interpolation_type"] = node.interpolation_type
            node_info["clamp"] = node.clamp
        elif node.type == "COLORRAMP" or node.type == "VALTORGB":
            cr = node.color_ramp
            node_info["color_ramp"] = {
                "interpolation": cr.interpolation,
                "color_mode": cr.color_mode,
                "elements": [
                    {"position": e.position, "color": list(e.color)}
                    for e in cr.elements
                ],
            }
        elif node.type == "GROUP":
            node_info["node_tree_name"] = node.node_tree.name if node.node_tree else None

        group_info["nodes"].append(node_info)

    for link in ng.links:
        group_info["links"].append({
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.name,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.name,
        })

    # Group interface inputs/outputs (Blender 4.x uses node_tree.interface)
    if hasattr(ng, "interface"):
        for item in ng.interface.items_tree:
            entry = {"name": item.name, "socket_type": item.socket_type if hasattr(item, "socket_type") else str(type(item))}
            if hasattr(item, "default_value"):
                dv = item.default_value
                try:
                    entry["default_value"] = list(dv)
                except TypeError:
                    entry["default_value"] = dv
            if hasattr(item, "in_out"):
                if item.in_out == "INPUT":
                    group_info["group_inputs"].append(entry)
                else:
                    group_info["group_outputs"].append(entry)

    groups.append(group_info)

# Also get world shader if present
world_results = []
if bpy.context.scene.world and bpy.context.scene.world.use_nodes:
    world = bpy.context.scene.world
    world_info = {
        "world": world.name,
        "nodes": [],
        "links": [],
    }
    for node in world.node_tree.nodes:
        node_info = {
            "name": node.name,
            "type": node.type,
            "bl_idname": node.bl_idname,
            "label": node.label,
            "inputs": {},
        }
        for inp in node.inputs:
            val = None
            if hasattr(inp, "default_value"):
                dv = inp.default_value
                try:
                    val = list(dv)
                except TypeError:
                    val = dv
            node_info["inputs"][inp.name] = {"type": inp.type, "default_value": val}
        world_info["nodes"].append(node_info)
    for link in world.node_tree.links:
        world_info["links"].append({
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.name,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.name,
        })
    world_results.append(world_info)

output = {
    "materials": results,
    "node_groups": groups,
    "world": world_results,
}

# Custom serializer for non-serializable types
def default_serializer(obj):
    if hasattr(obj, "__iter__"):
        return list(obj)
    return str(obj)

output_path = "/Users/jeantimex/Workspace/github/procedural-clouds/blend_parsed.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=default_serializer)

print(f"Written to {output_path}")
print(f"Materials: {len(results)}")
print(f"Node groups: {len(groups)}")
print(f"World shaders: {len(world_results)}")
