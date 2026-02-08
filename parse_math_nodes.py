"""Extract exact per-socket input values for all Math nodes in the cloud shader group."""
import bpy, json

ng = bpy.data.node_groups.get("Procedural Clouds Shader")
results = {}

for node in ng.nodes:
    if node.type in ("MATH", "VECT_MATH", "MAP_RANGE"):
        inputs = []
        for i, inp in enumerate(node.inputs):
            val = None
            if hasattr(inp, "default_value"):
                dv = inp.default_value
                try:
                    val = list(dv)
                except TypeError:
                    val = dv
            connected = any(
                link.to_node == node and link.to_socket.name == inp.name
                and list(node.inputs).index(link.to_socket) == i
                for link in ng.links
            )
            inputs.append({
                "index": i,
                "name": inp.name,
                "default": val,
                "connected": connected,
            })
        entry = {
            "type": node.type,
            "inputs": inputs,
        }
        if node.type == "MATH":
            entry["operation"] = node.operation
            entry["use_clamp"] = node.use_clamp
        elif node.type == "VECT_MATH":
            entry["operation"] = node.operation
        elif node.type == "MAP_RANGE":
            entry["clamp"] = node.clamp
        results[node.name] = entry

# Also get voronoi/noise node input details
for node in ng.nodes:
    if node.type in ("TEX_VORONOI", "TEX_NOISE"):
        inputs = []
        for i, inp in enumerate(node.inputs):
            val = None
            if hasattr(inp, "default_value"):
                dv = inp.default_value
                try:
                    val = list(dv)
                except TypeError:
                    val = dv
            connected = any(
                link.to_node == node and list(node.inputs).index(link.to_socket) == i
                for link in ng.links
            )
            inputs.append({
                "index": i,
                "name": inp.name,
                "default": val,
                "connected": connected,
            })
        entry = {"type": node.type, "inputs": inputs}
        if node.type == "TEX_VORONOI":
            entry["dimensions"] = node.voronoi_dimensions
            entry["feature"] = node.feature
            entry["distance"] = node.distance
        else:
            entry["dimensions"] = node.noise_dimensions
        results[node.name] = entry

with open("/Users/jeantimex/Workspace/github/procedural-clouds/math_nodes.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Extracted {len(results)} nodes")
