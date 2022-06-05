
import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil


class MainWindow(QTabWidget):
    # 初始化
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/panda.jpg'))
        self.setWindowTitle('猫咪分类')


        self.model = tf.keras.models.load_model("models/cats_cnn.h5")
        self.to_predict_name = "images/panda.jpg"
        self.class_names = ["亚洲豹猫", "加拿大无毛猫", "布偶猫", "暹罗猫", "橘猫", "熊猫", "狸花猫", "美国短毛猫", "英国短毛猫"]
        self.resize(12010, 1200)
        self.initUI()

    # 界面初始化，设置界面布局
    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('华文彩云', 25)

        # 主页面，设置组件并在组件放在布局上
        left_widget = QWidget()
        left_layout = QVBoxLayout()



        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/panda1.jpg"))
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 选择图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        self.result = QLabel("请上传图片")
        self.result.setFont(QFont('华文琥珀', 80))
        right_layout.addStretch()

        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        # 关于页面，设置组件并把组件放在布局上
        about_widget = QWidget()
        about_layout = QVBoxLayout()



        about_img = QLabel()

        about_img.setAlignment(Qt.AlignCenter)

        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()

        about_widget.setLayout(about_layout)

        # 添加注释
        self.addTab(main_widget, 'Cats')




    # 上传并显示图片
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')  # 打开文件选择框选择文件
        img_name = openfile_name[0]  # 获取图片名称
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmp_up." + img_name.split(".")[-1]  # 将图片移动到当前目录
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)  # 打开图片
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)  # 将图片的大小统一调整到400的高，方便界面显示
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))  # 将图片大小调整到224*224用于模型推理
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))
            self.result.setText("请上传图片")

    # 预测图片
    def predict_img(self):
        img = Image.open('images/target.png')  # 读取图片
        img = np.asarray(img)  # 将图片转化为numpy的数组
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))  # 将图片输入模型得到结果
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]  # 获得对应的水果名称
        self.result.setText(result)  # 在界面上做显示




if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
