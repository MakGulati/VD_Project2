class object():  # structure with list of SIFT keypoints per object with the object id
    def __init__(self, _obj_id, _des_mat):
        self.obj_id = _obj_id  # id of the object
        self.des_mat = _des_mat  # matrix of keypoints

    def __len__(self):
        return self.des_mat.shape[0]

    def get_des(self, _pos_id):
        return self.des_mat[_pos_id]

    def __del__(self):
        pass


class node(object):
    def __init__(self):
        self.name = None
        self.node = []
        self.otherInfo = None
        self.prev = None

    def nex(self, child):
        "Gets a node by number"
        return self.node[child]

    def prev(self):
        return self.prev

    def goto(self, data):
        "Gets the node by name"
        for child in range(0, len(self.node)):
            if self.node[child].name == data:
                return self.node[child]

    def add(self):
        node1 = node()
        self.node.append(node1)
        node1.prev = self
        return node1
