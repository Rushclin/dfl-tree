from flask import Flask, request, jsonify

app = Flask(__name__)

# Données du nœud
node_data = {
    "id": None,
    "parent": None,
    "left": None,
    "right": None,
}


@app.route("/init", methods=["POST"])
def init_node():
    """
    Initialiser un nœud avec un ID et des liens parent-gauche-droit.
    """
    global node_data
    data = request.get_json()
    node_data["id"] = data.get("id")
    node_data["parent"] = data.get("parent")
    node_data["left"] = data.get("left")
    node_data["right"] = data.get("right")
    return jsonify({"message": "Node initialized", "node": node_data}), 200


@app.route("/rotate", methods=["POST"])
def rotate_node():
    """
    Effectuer une rotation du nœud (gauche ou droite).
    """
    global node_data
    direction = request.get_json().get("direction")
    if direction == "left":
        temp = node_data["right"]
        node_data["right"] = node_data["id"]
        node_data["id"] = temp
    elif direction == "right":
        temp = node_data["left"]
        node_data["left"] = node_data["id"]
        node_data["id"] = temp
    else:
        return jsonify({"error": "Invalid direction"}), 400
    return jsonify({"message": "Node rotated", "node": node_data}), 200


@app.route("/info", methods=["GET"])
def get_info():
    """
    Obtenir les informations du nœud.
    """
    return jsonify(node_data), 200


@app.route("/update_child", methods=["POST"])
def update_child():
    """
    Mettre à jour un enfant (gauche ou droite).
    """
    global node_data
    data = request.get_json()
    position = data.get("position")
    child_id = data.get("child_id")

    if position == "left":
        node_data["left"] = child_id
    elif position == "right":
        node_data["right"] = child_id
    else:
        return jsonify({"error": "Invalid position"}), 400

    return jsonify({"message": "Child updated", "node": node_data}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
