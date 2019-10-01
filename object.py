class keypoints_mat_with_id():  # structure with matrix of SIFT keypoints per object with the document id
    def __init__(self, _des_mat, _doc_id):
        self.des_mat = _des_mat  # matrix of SIFT keypoints for that document
        self.doc_id = _doc_id  # document id of the object
        
    def __len__(self):
        return self.des_mat.shape[0]

    def get_des(self, _pos_id):
        return self.des_mat[_pos_id]

    def __del__(self):
        pass


class keypoint_with_id(): # structure with SIFT keypoint vector and corresponding document id
    def __init__(self, _vector, _id):
        self.vector = _vector # SIFT vector
        self.id = _id # document id of that keypoint

    def __del__(self):
        pass