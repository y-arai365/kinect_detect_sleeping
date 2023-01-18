import numpy as np
from scipy.spatial import distance


class MotionDetector:
    def __init__(self, motion_point=0, sleep_point=0, frame_number=50):
        self.motion_point = motion_point
        self.sleep_point = sleep_point

        self.array_frames = np.zeros((frame_number, 3), dtype=float)

    @staticmethod
    def load(npy_file):
        """npyファイルのロード"""
        return np.load(npy_file)  # (フレーム数, 関節数, 各ポイント)

    @staticmethod
    def get_parts_array(array, parts_number):
        """各関節毎の配列を取得(各フレームのxyzを取得)"""
        return array[:, parts_number][:, :3]

    def motion_detect(self, arr):
        """前のフレームと比較して、どれくらい移動があるかを計算、全フレームにおいて移動があった回数を返す"""
        self.motion_point = 0
        for i, pt1 in enumerate(arr):
            pt2 = arr[max(i - 1, 0)]
            self._get_distance_between_two_points(pt1, pt2)
        return self.motion_point

    def get_weighted_moving_average_array(self, arr, weight):  # arr=(500, 3)
        """50フレームの配列取得と50フレームで加重移動平均を求めることを全フレームで繰り返して配列を取得"""
        xyz = []
        for coordinate in arr:
            self.update_array_50_frames(coordinate)
            self._get_weighted_moving_average(self.array_frames, weight, xyz)
        return xyz

    def update_array_50_frames(self, new_arr):  # new_arr=(3,)
        """現状のフレームの１番古いものと新たなものを入れ替える"""
        self.array_frames = np.roll(self.array_frames, -1, axis=0)
        self.array_frames[-1] = new_arr

    def _get_distance_between_two_points(self, pt1, pt2, distance_thresh=1):
        """2点間の距離を求める(3次元)、移動量が一定以上なら動いていると判定"""  # sleep_pointは後付け
        dist = distance.euclidean(pt1, pt2)
        if dist > distance_thresh:
            self.motion_point += 1
            self.sleep_point = 0
        else:
            self.sleep_point += 1
            if self.sleep_point > 30:
                # print("sleep_alarm=========="*10)
                pass

    @staticmethod
    def _get_weighted_moving_average(arr, weight, xyz):
        """xyz各方向の加重移動平均を求め、それをリストに入れる"""
        arr_x = arr[:, 0]
        arr_y = arr[:, 1]
        arr_z = arr[:, 2]
        arr_x_w = np.average(arr_x, weights=weight)
        arr_y_w = np.average(arr_y, weights=weight)
        arr_z_w = np.average(arr_z, weights=weight)
        xyz.append([arr_x_w, arr_y_w, arr_z_w])


if __name__ == '__main__':
    md = MotionDetector()

    a = md.load(r"numpy_file/sleeping.npy")
    b = md.load(r"numpy_file/eating.npy")
    parts_number_ = 31
    arr_a_ = md.get_parts_array(a, parts_number_)
    arr_b_ = md.get_parts_array(b, parts_number_)

    weight_ = np.ones(50, dtype="int8")
    weight_[10:20], weight_[20:30], weight_[30:40], weight_[40:50] = 5, 25, 50, 100

    xyz_a_ = md.get_weighted_moving_average_array(arr_a_, weight_)
    motion_a_ = md.motion_detect(xyz_a_)
    print("====================="*5)
    xyz_b_ = md.get_weighted_moving_average_array(arr_b_, weight_)
    motion_b_ = md.motion_detect(xyz_b_)
    print("sleeping.npy move:", motion_a_ / len(xyz_a_) * 100, "%")
    print("eating.npy move:", motion_b_ / len(xyz_b_) * 100, "%")

    """
    動いてるかどうかは判定可能
    寝ているかどうかはどう判定？
    ・起きてる人・寝ている人の動きの収集、機械学習で分類
    ・動いていないはただの一要素、他の要素(瞼の閉じ具合とか、体温の変化とか)と合わせて判定
    ・作業の進行度で判定(個人の作業速度に対して、どれだけ遅れがあるか)
    。。。
    """
