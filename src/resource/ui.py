# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'armlab_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1750, 1048)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.OutputFrame = QtWidgets.QFrame(self.centralwidget)
        self.OutputFrame.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OutputFrame.sizePolicy().hasHeightForWidth())
        self.OutputFrame.setSizePolicy(sizePolicy)
        self.OutputFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.OutputFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.OutputFrame.setObjectName("OutputFrame")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.OutputFrame)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.JointCoordLabel = QtWidgets.QLabel(self.OutputFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.JointCoordLabel.sizePolicy().hasHeightForWidth())
        self.JointCoordLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.JointCoordLabel.setFont(font)
        self.JointCoordLabel.setObjectName("JointCoordLabel")
        self.verticalLayout_5.addWidget(self.JointCoordLabel)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.BLabel = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.BLabel.setFont(font)
        self.BLabel.setObjectName("BLabel")
        self.verticalLayout_9.addWidget(self.BLabel, 0, QtCore.Qt.AlignRight)
        self.SLabel = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.SLabel.setFont(font)
        self.SLabel.setObjectName("SLabel")
        self.verticalLayout_9.addWidget(self.SLabel, 0, QtCore.Qt.AlignRight)
        self.ELabel = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.ELabel.setFont(font)
        self.ELabel.setObjectName("ELabel")
        self.verticalLayout_9.addWidget(self.ELabel, 0, QtCore.Qt.AlignRight)
        self.WALabel = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.WALabel.setFont(font)
        self.WALabel.setObjectName("WALabel")
        self.verticalLayout_9.addWidget(self.WALabel, 0, QtCore.Qt.AlignRight)
        self.WRLabel = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.WRLabel.setFont(font)
        self.WRLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.WRLabel.setObjectName("WRLabel")
        self.verticalLayout_9.addWidget(self.WRLabel)
        self.horizontalLayout_3.addLayout(self.verticalLayout_9)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.rdoutBaseJC = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutBaseJC.setFont(font)
        self.rdoutBaseJC.setObjectName("rdoutBaseJC")
        self.verticalLayout_8.addWidget(self.rdoutBaseJC, 0, QtCore.Qt.AlignLeft)
        self.rdoutShoulderJC = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutShoulderJC.setFont(font)
        self.rdoutShoulderJC.setObjectName("rdoutShoulderJC")
        self.verticalLayout_8.addWidget(self.rdoutShoulderJC, 0, QtCore.Qt.AlignLeft)
        self.rdoutElbowJC = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutElbowJC.setFont(font)
        self.rdoutElbowJC.setObjectName("rdoutElbowJC")
        self.verticalLayout_8.addWidget(self.rdoutElbowJC, 0, QtCore.Qt.AlignLeft)
        self.rdoutWristAJC = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutWristAJC.setFont(font)
        self.rdoutWristAJC.setObjectName("rdoutWristAJC")
        self.verticalLayout_8.addWidget(self.rdoutWristAJC, 0, QtCore.Qt.AlignLeft)
        self.rdoutWristRJC = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutWristRJC.setFont(font)
        self.rdoutWristRJC.setObjectName("rdoutWristRJC")
        self.verticalLayout_8.addWidget(self.rdoutWristRJC)
        self.horizontalLayout_3.addLayout(self.verticalLayout_8)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.WorldCoordLabel = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.WorldCoordLabel.setFont(font)
        self.WorldCoordLabel.setObjectName("WorldCoordLabel")
        self.verticalLayout_5.addWidget(self.WorldCoordLabel)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.XLabel = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.XLabel.setFont(font)
        self.XLabel.setScaledContents(False)
        self.XLabel.setObjectName("XLabel")
        self.verticalLayout_13.addWidget(self.XLabel, 0, QtCore.Qt.AlignRight)
        self.YLabel = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.YLabel.setFont(font)
        self.YLabel.setObjectName("YLabel")
        self.verticalLayout_13.addWidget(self.YLabel, 0, QtCore.Qt.AlignRight)
        self.ZLabel = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.ZLabel.setFont(font)
        self.ZLabel.setObjectName("ZLabel")
        self.verticalLayout_13.addWidget(self.ZLabel, 0, QtCore.Qt.AlignRight)
        self.PhiLabel = QtWidgets.QLabel(self.OutputFrame)
        self.PhiLabel.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.PhiLabel.setFont(font)
        self.PhiLabel.setObjectName("PhiLabel")
        self.verticalLayout_13.addWidget(self.PhiLabel, 0, QtCore.Qt.AlignRight)
        self.ThetaLabel = QtWidgets.QLabel(self.OutputFrame)
        self.ThetaLabel.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.ThetaLabel.setFont(font)
        self.ThetaLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.ThetaLabel.setObjectName("ThetaLabel")
        self.verticalLayout_13.addWidget(self.ThetaLabel)
        self.PsiLabel = QtWidgets.QLabel(self.OutputFrame)
        self.PsiLabel.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.PsiLabel.setFont(font)
        self.PsiLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.PsiLabel.setObjectName("PsiLabel")
        self.verticalLayout_13.addWidget(self.PsiLabel)
        self.horizontalLayout_4.addLayout(self.verticalLayout_13)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.rdoutX = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutX.setFont(font)
        self.rdoutX.setObjectName("rdoutX")
        self.verticalLayout_12.addWidget(self.rdoutX, 0, QtCore.Qt.AlignLeft)
        self.rdoutY = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutY.setFont(font)
        self.rdoutY.setObjectName("rdoutY")
        self.verticalLayout_12.addWidget(self.rdoutY, 0, QtCore.Qt.AlignLeft)
        self.rdoutZ = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutZ.setFont(font)
        self.rdoutZ.setObjectName("rdoutZ")
        self.verticalLayout_12.addWidget(self.rdoutZ, 0, QtCore.Qt.AlignLeft)
        self.rdoutPhi = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutPhi.setFont(font)
        self.rdoutPhi.setObjectName("rdoutPhi")
        self.verticalLayout_12.addWidget(self.rdoutPhi, 0, QtCore.Qt.AlignLeft)
        self.rdoutTheta = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutTheta.setFont(font)
        self.rdoutTheta.setObjectName("rdoutTheta")
        self.verticalLayout_12.addWidget(self.rdoutTheta)
        self.rdoutPsi = QtWidgets.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        self.rdoutPsi.setFont(font)
        self.rdoutPsi.setObjectName("rdoutPsi")
        self.verticalLayout_12.addWidget(self.rdoutPsi)
        self.horizontalLayout_4.addLayout(self.verticalLayout_12)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.Group2 = QtWidgets.QVBoxLayout()
        self.Group2.setContentsMargins(10, 10, 10, 10)
        self.Group2.setObjectName("Group2")
        self.btnUser1 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser1.setObjectName("btnUser1")
        self.Group2.addWidget(self.btnUser1)
        self.btnUser2 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser2.setObjectName("btnUser2")
        self.Group2.addWidget(self.btnUser2)
        self.btnUser3 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser3.setObjectName("btnUser3")
        self.Group2.addWidget(self.btnUser3)
        self.btnUser4 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser4.setObjectName("btnUser4")
        self.Group2.addWidget(self.btnUser4)
        self.btnUser5 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser5.setObjectName("btnUser5")
        self.Group2.addWidget(self.btnUser5)
        self.btnUser6 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser6.setObjectName("btnUser6")
        self.Group2.addWidget(self.btnUser6)
        self.btnUser7 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser7.setObjectName("btnUser7")
        self.Group2.addWidget(self.btnUser7)
        self.btnUser8 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser8.setObjectName("btnUser8")
        self.Group2.addWidget(self.btnUser8)
        self.btnUser9 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser9.setObjectName("btnUser9")
        self.Group2.addWidget(self.btnUser9)
        self.btnUser10 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser10.setAutoRepeatDelay(300)
        self.btnUser10.setObjectName("btnUser10")
        self.Group2.addWidget(self.btnUser10)
        self.btnUser11 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser11.setAutoRepeatDelay(300)
        self.btnUser11.setObjectName("btnUser11")
        self.Group2.addWidget(self.btnUser11)
        self.btnUser12 = QtWidgets.QPushButton(self.OutputFrame)
        self.btnUser12.setAutoRepeatDelay(300)
        self.btnUser12.setObjectName("btnUser12")
        self.Group2.addWidget(self.btnUser12)
        self.verticalLayout_5.addLayout(self.Group2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem)
        self.horizontalLayout_13.addWidget(self.OutputFrame)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem1)
        self.videoDisplay = QtWidgets.QLabel(self.centralwidget)
        self.videoDisplay.setMinimumSize(QtCore.QSize(1280, 720))
        self.videoDisplay.setMaximumSize(QtCore.QSize(1280, 720))
        self.videoDisplay.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.videoDisplay.setMouseTracking(True)
        self.videoDisplay.setFrameShape(QtWidgets.QFrame.Box)
        self.videoDisplay.setObjectName("videoDisplay")
        self.horizontalLayout_12.addWidget(self.videoDisplay)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.chk_directcontrol = QtWidgets.QCheckBox(self.centralwidget)
        self.chk_directcontrol.setChecked(False)
        self.chk_directcontrol.setObjectName("chk_directcontrol")
        self.horizontalLayout_2.addWidget(self.chk_directcontrol)
        self.radioVideo = QtWidgets.QRadioButton(self.centralwidget)
        self.radioVideo.setChecked(True)
        self.radioVideo.setAutoExclusive(True)
        self.radioVideo.setObjectName("radioVideo")
        self.horizontalLayout_2.addWidget(self.radioVideo)
        self.radioDepth = QtWidgets.QRadioButton(self.centralwidget)
        self.radioDepth.setObjectName("radioDepth")
        self.horizontalLayout_2.addWidget(self.radioDepth)
        self.radioUsr1 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioUsr1.setObjectName("radioUsr1")
        self.horizontalLayout_2.addWidget(self.radioUsr1)
        self.radioUsr2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioUsr2.setObjectName("radioUsr2")
        self.horizontalLayout_2.addWidget(self.radioUsr2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.PixelCoordLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PixelCoordLabel.setFont(font)
        self.PixelCoordLabel.setObjectName("PixelCoordLabel")
        self.horizontalLayout_2.addWidget(self.PixelCoordLabel)
        self.rdoutMousePixels = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.rdoutMousePixels.setFont(font)
        self.rdoutMousePixels.setTextFormat(QtCore.Qt.AutoText)
        self.rdoutMousePixels.setObjectName("rdoutMousePixels")
        self.horizontalLayout_2.addWidget(self.rdoutMousePixels)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.PixelCoordLabel_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PixelCoordLabel_2.setFont(font)
        self.PixelCoordLabel_2.setObjectName("PixelCoordLabel_2")
        self.horizontalLayout_2.addWidget(self.PixelCoordLabel_2)
        self.rdoutMouseWorld = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Mono")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.rdoutMouseWorld.setFont(font)
        self.rdoutMouseWorld.setTextFormat(QtCore.Qt.AutoText)
        self.rdoutMouseWorld.setObjectName("rdoutMouseWorld")
        self.horizontalLayout_2.addWidget(self.rdoutMouseWorld)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.SliderFrame = QtWidgets.QFrame(self.centralwidget)
        self.SliderFrame.setEnabled(False)
        self.SliderFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.SliderFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.SliderFrame.setLineWidth(1)
        self.SliderFrame.setObjectName("SliderFrame")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.SliderFrame)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_16.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.BLabelS = QtWidgets.QLabel(self.SliderFrame)
        self.BLabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.BLabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.BLabelS.setObjectName("BLabelS")
        self.horizontalLayout.addWidget(self.BLabelS)
        self.sldrBase = QtWidgets.QSlider(self.SliderFrame)
        self.sldrBase.setMinimum(-179)
        self.sldrBase.setMaximum(180)
        self.sldrBase.setOrientation(QtCore.Qt.Horizontal)
        self.sldrBase.setObjectName("sldrBase")
        self.horizontalLayout.addWidget(self.sldrBase)
        self.rdoutBase = QtWidgets.QLabel(self.SliderFrame)
        self.rdoutBase.setMinimumSize(QtCore.QSize(30, 0))
        self.rdoutBase.setObjectName("rdoutBase")
        self.horizontalLayout.addWidget(self.rdoutBase)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.SLabelS = QtWidgets.QLabel(self.SliderFrame)
        self.SLabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.SLabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.SLabelS.setObjectName("SLabelS")
        self.horizontalLayout_7.addWidget(self.SLabelS)
        self.sldrShoulder = QtWidgets.QSlider(self.SliderFrame)
        self.sldrShoulder.setMinimum(-179)
        self.sldrShoulder.setMaximum(180)
        self.sldrShoulder.setOrientation(QtCore.Qt.Horizontal)
        self.sldrShoulder.setObjectName("sldrShoulder")
        self.horizontalLayout_7.addWidget(self.sldrShoulder)
        self.rdoutShoulder = QtWidgets.QLabel(self.SliderFrame)
        self.rdoutShoulder.setObjectName("rdoutShoulder")
        self.horizontalLayout_7.addWidget(self.rdoutShoulder)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.ELabelS = QtWidgets.QLabel(self.SliderFrame)
        self.ELabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.ELabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.ELabelS.setObjectName("ELabelS")
        self.horizontalLayout_8.addWidget(self.ELabelS)
        self.sldrElbow = QtWidgets.QSlider(self.SliderFrame)
        self.sldrElbow.setMinimum(-179)
        self.sldrElbow.setMaximum(180)
        self.sldrElbow.setOrientation(QtCore.Qt.Horizontal)
        self.sldrElbow.setObjectName("sldrElbow")
        self.horizontalLayout_8.addWidget(self.sldrElbow)
        self.rdoutElbow = QtWidgets.QLabel(self.SliderFrame)
        self.rdoutElbow.setObjectName("rdoutElbow")
        self.horizontalLayout_8.addWidget(self.rdoutElbow)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.WALabelS = QtWidgets.QLabel(self.SliderFrame)
        self.WALabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.WALabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.WALabelS.setObjectName("WALabelS")
        self.horizontalLayout_11.addWidget(self.WALabelS)
        self.sldrWristA = QtWidgets.QSlider(self.SliderFrame)
        self.sldrWristA.setMinimum(-179)
        self.sldrWristA.setMaximum(180)
        self.sldrWristA.setOrientation(QtCore.Qt.Horizontal)
        self.sldrWristA.setObjectName("sldrWristA")
        self.horizontalLayout_11.addWidget(self.sldrWristA)
        self.rdoutWristA = QtWidgets.QLabel(self.SliderFrame)
        self.rdoutWristA.setObjectName("rdoutWristA")
        self.horizontalLayout_11.addWidget(self.rdoutWristA)
        self.verticalLayout_2.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.WRLabelS = QtWidgets.QLabel(self.SliderFrame)
        self.WRLabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.WRLabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.WRLabelS.setObjectName("WRLabelS")
        self.horizontalLayout_15.addWidget(self.WRLabelS)
        self.sldrWristR = QtWidgets.QSlider(self.SliderFrame)
        self.sldrWristR.setMinimum(-179)
        self.sldrWristR.setMaximum(180)
        self.sldrWristR.setOrientation(QtCore.Qt.Horizontal)
        self.sldrWristR.setObjectName("sldrWristR")
        self.horizontalLayout_15.addWidget(self.sldrWristR)
        self.rdoutWristR = QtWidgets.QLabel(self.SliderFrame)
        self.rdoutWristR.setObjectName("rdoutWristR")
        self.horizontalLayout_15.addWidget(self.rdoutWristR)
        self.verticalLayout_2.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_9.addLayout(self.verticalLayout_2)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.MoveTimeLabel = QtWidgets.QLabel(self.SliderFrame)
        self.MoveTimeLabel.setObjectName("MoveTimeLabel")
        self.verticalLayout_10.addWidget(self.MoveTimeLabel)
        self.sldrMoveTime = QtWidgets.QSlider(self.SliderFrame)
        self.sldrMoveTime.setMaximum(100)
        self.sldrMoveTime.setProperty("value", 30)
        self.sldrMoveTime.setOrientation(QtCore.Qt.Vertical)
        self.sldrMoveTime.setObjectName("sldrMoveTime")
        self.verticalLayout_10.addWidget(self.sldrMoveTime)
        self.rdoutMoveTime = QtWidgets.QLabel(self.SliderFrame)
        self.rdoutMoveTime.setObjectName("rdoutMoveTime")
        self.verticalLayout_10.addWidget(self.rdoutMoveTime)
        self.horizontalLayout_9.addLayout(self.verticalLayout_10)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.AccelTimeLabel = QtWidgets.QLabel(self.SliderFrame)
        self.AccelTimeLabel.setObjectName("AccelTimeLabel")
        self.verticalLayout_7.addWidget(self.AccelTimeLabel)
        self.sldrAccelTime = QtWidgets.QSlider(self.SliderFrame)
        self.sldrAccelTime.setMaximum(100)
        self.sldrAccelTime.setProperty("value", 20)
        self.sldrAccelTime.setSliderPosition(20)
        self.sldrAccelTime.setOrientation(QtCore.Qt.Vertical)
        self.sldrAccelTime.setObjectName("sldrAccelTime")
        self.verticalLayout_7.addWidget(self.sldrAccelTime)
        self.rdoutAccelTime = QtWidgets.QLabel(self.SliderFrame)
        self.rdoutAccelTime.setObjectName("rdoutAccelTime")
        self.verticalLayout_7.addWidget(self.rdoutAccelTime)
        self.horizontalLayout_9.addLayout(self.verticalLayout_7)
        self.verticalLayout_16.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.verticalLayout_16.addLayout(self.horizontalLayout_10)
        self.verticalLayout_4.addWidget(self.SliderFrame)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setMaximumSize(QtCore.QSize(125, 16777215))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_14.addWidget(self.label_3)
        self.rdoutStatus = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.rdoutStatus.setFont(font)
        self.rdoutStatus.setTextFormat(QtCore.Qt.AutoText)
        self.rdoutStatus.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.rdoutStatus.setWordWrap(True)
        self.rdoutStatus.setObjectName("rdoutStatus")
        self.horizontalLayout_14.addWidget(self.rdoutStatus)
        self.verticalLayout_4.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_13.addLayout(self.verticalLayout_4)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.btn_estop = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_estop.setFont(font)
        self.btn_estop.setObjectName("btn_estop")
        self.verticalLayout_11.addWidget(self.btn_estop)
        self.btn_init_arm = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_init_arm.setFont(font)
        self.btn_init_arm.setObjectName("btn_init_arm")
        self.verticalLayout_11.addWidget(self.btn_init_arm)
        self.btn_sleep_arm = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_sleep_arm.setFont(font)
        self.btn_sleep_arm.setObjectName("btn_sleep_arm")
        self.verticalLayout_11.addWidget(self.btn_sleep_arm)
        self.btn_torq_off = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_torq_off.setFont(font)
        self.btn_torq_off.setObjectName("btn_torq_off")
        self.verticalLayout_11.addWidget(self.btn_torq_off)
        self.btn_torq_on = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_torq_on.setFont(font)
        self.btn_torq_on.setObjectName("btn_torq_on")
        self.verticalLayout_11.addWidget(self.btn_torq_on)
        self.btn_calibrate = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_calibrate.setFont(font)
        self.btn_calibrate.setObjectName("btn_calibrate")
        self.verticalLayout_11.addWidget(self.btn_calibrate)
        self.btn_task1 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task1.setFont(font)
        self.btn_task1.setObjectName("btn_task1")
        self.verticalLayout_11.addWidget(self.btn_task1)
        self.btn_task2 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task2.setFont(font)
        self.btn_task2.setObjectName("btn_task2")
        self.verticalLayout_11.addWidget(self.btn_task2)
        self.btn_task3 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task3.setFont(font)
        self.btn_task3.setObjectName("btn_task3")
        self.verticalLayout_11.addWidget(self.btn_task3)
        self.btn_task4 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task4.setFont(font)
        self.btn_task4.setObjectName("btn_task4")
        self.verticalLayout_11.addWidget(self.btn_task4)
        self.btn_task5 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task5.setFont(font)
        self.btn_task5.setObjectName("btn_task5")
        self.verticalLayout_11.addWidget(self.btn_task5)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_11.addItem(spacerItem6)
        self.horizontalLayout_13.addLayout(self.verticalLayout_11)
        self.verticalLayout_14.addLayout(self.horizontalLayout_13)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.JointCoordLabel.setText(_translate("MainWindow", "Joint Coordinates"))
        self.BLabel.setText(_translate("MainWindow", "B:"))
        self.SLabel.setText(_translate("MainWindow", "S:"))
        self.ELabel.setText(_translate("MainWindow", "E:"))
        self.WALabel.setText(_translate("MainWindow", "WA:"))
        self.WRLabel.setText(_translate("MainWindow", "WR:"))
        self.rdoutBaseJC.setText(_translate("MainWindow", "0"))
        self.rdoutShoulderJC.setText(_translate("MainWindow", "0"))
        self.rdoutElbowJC.setText(_translate("MainWindow", "0"))
        self.rdoutWristAJC.setText(_translate("MainWindow", "0"))
        self.rdoutWristRJC.setText(_translate("MainWindow", "0"))
        self.WorldCoordLabel.setText(_translate("MainWindow", "End Effector Location"))
        self.XLabel.setText(_translate("MainWindow", "X:"))
        self.YLabel.setText(_translate("MainWindow", "Y:"))
        self.ZLabel.setText(_translate("MainWindow", "Z:"))
        self.PhiLabel.setText(_translate("MainWindow", "Phi:"))
        self.ThetaLabel.setText(_translate("MainWindow", "Theta:"))
        self.PsiLabel.setText(_translate("MainWindow", "Psi:"))
        self.rdoutX.setText(_translate("MainWindow", "0"))
        self.rdoutY.setText(_translate("MainWindow", "0"))
        self.rdoutZ.setText(_translate("MainWindow", "0"))
        self.rdoutPhi.setText(_translate("MainWindow", "0"))
        self.rdoutTheta.setText(_translate("MainWindow", "0"))
        self.rdoutPsi.setText(_translate("MainWindow", "0"))
        self.btnUser1.setText(_translate("MainWindow", "USER 1"))
        self.btnUser2.setText(_translate("MainWindow", "USER 2"))
        self.btnUser3.setText(_translate("MainWindow", "USER 3"))
        self.btnUser4.setText(_translate("MainWindow", "USER 4"))
        self.btnUser5.setText(_translate("MainWindow", "USER 5"))
        self.btnUser6.setText(_translate("MainWindow", "USER 6"))
        self.btnUser7.setText(_translate("MainWindow", "USER 7"))
        self.btnUser8.setText(_translate("MainWindow", "USER 8"))
        self.btnUser9.setText(_translate("MainWindow", "USER 9"))
        self.btnUser10.setText(_translate("MainWindow", "USER 10"))
        self.btnUser11.setText(_translate("MainWindow", "USER 11"))
        self.btnUser12.setText(_translate("MainWindow", "USER 12"))
        self.videoDisplay.setText(_translate("MainWindow", "Video Display"))
        self.chk_directcontrol.setText(_translate("MainWindow", "Direct Control"))
        self.radioVideo.setText(_translate("MainWindow", "Video"))
        self.radioDepth.setText(_translate("MainWindow", "Depth"))
        self.radioUsr1.setText(_translate("MainWindow", "Tags"))
        self.radioUsr2.setText(_translate("MainWindow", "User 2"))
        self.PixelCoordLabel.setText(_translate("MainWindow", "Mouse Coordinates:"))
        self.rdoutMousePixels.setText(_translate("MainWindow", "(U,V,D)"))
        self.PixelCoordLabel_2.setText(_translate("MainWindow", "World Coordinates [mm]:"))
        self.rdoutMouseWorld.setText(_translate("MainWindow", "(X,Y,Z)"))
        self.BLabelS.setText(_translate("MainWindow", "Base"))
        self.rdoutBase.setText(_translate("MainWindow", "0"))
        self.SLabelS.setText(_translate("MainWindow", "Shoulder"))
        self.rdoutShoulder.setText(_translate("MainWindow", "0"))
        self.ELabelS.setText(_translate("MainWindow", "Elbow"))
        self.rdoutElbow.setText(_translate("MainWindow", "0"))
        self.WALabelS.setText(_translate("MainWindow", "Wrist Angle"))
        self.rdoutWristA.setText(_translate("MainWindow", "0"))
        self.WRLabelS.setText(_translate("MainWindow", "Wrist Rotate"))
        self.rdoutWristR.setText(_translate("MainWindow", "0"))
        self.MoveTimeLabel.setText(_translate("MainWindow", "MoveTime"))
        self.rdoutMoveTime.setText(_translate("MainWindow", "0"))
        self.AccelTimeLabel.setText(_translate("MainWindow", "AccelTime"))
        self.rdoutAccelTime.setText(_translate("MainWindow", "0"))
        self.label_3.setText(_translate("MainWindow", "Status:"))
        self.rdoutStatus.setText(_translate("MainWindow", "Waiting for Inputs"))
        self.btn_estop.setText(_translate("MainWindow", "EMERGENCY STOP"))
        self.btn_init_arm.setText(_translate("MainWindow", "INITIALIZE ARM"))
        self.btn_sleep_arm.setText(_translate("MainWindow", "SLEEP ARM"))
        self.btn_torq_off.setText(_translate("MainWindow", "TORQUE OFF"))
        self.btn_torq_on.setText(_translate("MainWindow", "TORQUE ON"))
        self.btn_calibrate.setText(_translate("MainWindow", "CALIBRATE"))
        self.btn_task1.setText(_translate("MainWindow", "TASK 1"))
        self.btn_task2.setText(_translate("MainWindow", "TASK 2"))
        self.btn_task3.setText(_translate("MainWindow", "TASK 3"))
        self.btn_task4.setText(_translate("MainWindow", "TASK 4"))
        self.btn_task5.setText(_translate("MainWindow", "TASK 5"))
