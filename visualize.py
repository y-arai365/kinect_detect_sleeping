import cv2
import numpy as np
import open3d as o3d


class Renderer:
    def __init__(self, point_size=10):
        """
        Open3Dのvisualization.Visualizer()で3次元空間上に配置したジオメトリーを画像に変換。
        """
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(visible=False)
        self._ctrl = self._vis.get_view_control()
        self._vis.get_render_option().point_size = point_size

    def close(self):
        """終了処理"""
        self._vis.close()

    def rotate(self, x, y):
        """
        ジオメトリーを回転させる

        Args:
            x (float): 横方向に回転させる量
            y (float): 縦方向に回転させる量
        """
        self._ctrl.rotate(x, y, 0, 0)

    def scale(self, ratio):
        """
        ジオメトリーを拡大縮小する

        Args:
            ratio (float):　拡大縮小率
        """
        self._ctrl.scale(ratio)

    def translate(self, x, y):
        """
        ジオメトリーを上下左右に移動させる。

        Args:
            x (float): 横方向の移動量
            y (float):　縦方向の移動量
        """
        self._ctrl.translate(x, y)

    def to_image(self, geometries, convert_color=False):
        """
        ジオメトリーを渡し、それを３D空間上に配置した画像を返す。

        Args:
            geometries (list[open3d.geometry.Geometry]): 点群、メッシュなどのジオメトリーのリスト
            convert_color (bool): RGBのままか、OpenCV用にBGRにするか。

        Returns:
            img: 3チャンネル画像

        """
        self._show_geometries(geometries)
        self._vis.poll_events()
        rendered_image = self._vis.capture_screen_float_buffer(False)
        img_rgb = np.uint8(np.array(rendered_image) * 255)
        if convert_color:
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            return img_rgb

    def _add_geometry(self, geometries):
        for geometry in geometries:
            self._vis.add_geometry(geometry)
        self._show_geometries = self._update_geometry  # 2回目以降はupdate_geometry

    def _update_geometry(self, geometries):
        for geometry in geometries:
            self._vis.update_geometry(geometry)

    def _show_geometries(self, geometries):
        self._add_geometry(geometries)  # 初回だけadd_geometry


render = Renderer()


def mouse_event(event, x, y, flag, _param):
    # 回転
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_event.lbutton_down = True
        mouse_event.x_temp, mouse_event.y_temp = x, y
    elif event == cv2.EVENT_MOUSEMOVE and mouse_event.lbutton_down:
        render.rotate(x - mouse_event.x_temp, y - mouse_event.y_temp)
        mouse_event.x_temp, mouse_event.y_temp = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_event.lbutton_down = False
    # 移動
    elif event == cv2.EVENT_RBUTTONDOWN:
        mouse_event.rbutton_down = True
        mouse_event.x_temp, mouse_event.y_temp = x, y
    elif event == cv2.EVENT_MOUSEMOVE and mouse_event.rbutton_down:
        render.translate(x - mouse_event.x_temp, y - mouse_event.y_temp)
        mouse_event.x_temp, mouse_event.y_temp = x, y
    elif event == cv2.EVENT_RBUTTONUP:
        mouse_event.rbutton_down = False
    # 拡大縮小
    elif event == cv2.EVENT_MOUSEWHEEL:
        render.scale(np.sign(flag) * 5)


mouse_event.lbutton_down = False
mouse_event.rbutton_down = False

points = np.load(r"numpy_file/eating.npy")

pcd = o3d.geometry.PointCloud()
lineset = o3d.geometry.LineSet()

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("img", mouse_event)


def line(points, lineset):
    lines = [[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [5, 6], [6, 7],
             [8, 7], [9, 8], [10, 7], [11, 2], [12, 11], [13, 12], [14, 13], [15, 14],
             [16, 15], [17, 14], [18, 0], [19, 18], [20, 19], [21, 20], [22, 0], [23, 22],
             [24, 23], [25, 24], [26, 3], [27, 26], [28, 26], [29, 26], [30, 26], [31, 26]]

    lineset.points = points
    lineset.lines = o3d.utility.Vector2iVector(lines)


i = 0
while i < len(points):
    pcd.points = o3d.utility.Vector3dVector(points[i, :, :3])
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    line(pcd.points, lineset)

    img = render.to_image([pcd, lineset])
    cv2.imshow("img", img)

    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    i += 1

cv2.destroyAllWindows()
render.close()
