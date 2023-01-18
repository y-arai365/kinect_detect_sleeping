import numpy as np
from scipy.spatial import distance


class SleepDetector:
    def __init__(self, sleep_point=0):
        self.motion_point = sleep_point

    @staticmethod
    def load(npy_file):
        """npyファイルのロード"""
        return np.load(npy_file)  # (フレーム数, 関節数, 各ポイント)

    @staticmethod
    def get_parts_array(array, parts_number):
        """各関節毎の配列を取得(各フレームのxyzを取得)"""
        return array[:, parts_number][:, :3]

    def get_distance_between_two_points(self, pt1, pt2):
        """2点間の距離を求める(3次元)、移動量が一定以上なら動いていると判定"""
        dist = distance.euclidean(pt1, pt2)
        if dist > 5:
            self.motion_point += 1

    def motion_detect(self, arr):
        """前のフレームと比較して、どれくらい移動があるかを計算、全フレームにおいて移動があった回数を返す"""
        self.motion_point = 0
        for i, pt1 in enumerate(arr):
            pt2 = arr[max(i - 1, 0)]
            self.get_distance_between_two_points(pt1, pt2)
        return self.motion_point


if __name__ == '__main__':
    sd = SleepDetector()

    a = sd.load(r"numpy_file/sleeping_.npy")
    b = sd.load(r"numpy_file/reading_.npy")
    parts_number_ = 21

    arr_a_ = sd.get_parts_array(a, parts_number_)
    arr_b_ = sd.get_parts_array(b, parts_number_)

    motion_a_ = sd.motion_detect(arr_a_)
    motion_b_ = sd.motion_detect(arr_b_)
    print("sleeping.npy move:", motion_a_ / a.shape[0] * 100, "%")
    print("reading.npy move:", motion_b_ / b.shape[0] * 100, "%")
