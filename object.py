class object():  # structure with list of SIFT keypoints per object with the object id
    def __init__(self, _obj_id, _des_mat):
        self.obj_id = _obj_id # id of the object
        self.des_mat = _des_mat # matrix of keypoints

    def __len__(self):
        return self.des_mat.shape[0]

    def get_des(self, _pos_id):
        return self.des_mat[_pos_id]

    def __del__(self):
        pass
