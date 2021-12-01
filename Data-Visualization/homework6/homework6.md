### 可视化作业 6

##### 张言健 16300200020

##### 环境配置

```
VTK 9.0.0
python 3.6
Windows 10
```

1 阅读了解VTK（VTK - The Visualization Toolkit, www.vtk.org），学习某个编程环境下调用VTK库进行可视化。该题只需要回答已经学习（完成作业）或没有学习（未完成作业）。

##### 已经学习，完成作业。

2 调用可视化渲染引擎库（如VTK），实现三维体数据完整的渲染过程（如光照模型，颜色设置等）。需要实现的渲染过程包括：(1) 等值面渲染，(2) 体渲染。**请自己找一个体数据进行测试和结果展示**。提交作业需要对使用数据进行说明，并提交源数据（或数据下载的网上链接）。

##### pipline示意图：

![pipeline](.\image\pipeline.PNG)

过程详见代码

##### 等值面渲染结果

| 正视图                              | 侧视图                            | 俯视图                            |
| ----------------------------------- | --------------------------------- | --------------------------------- |
| ![SUR-front](.\image\SUR-front.PNG) | ![SUR-side](.\image\SUR-side.PNG) | ![SUR-down](.\image\SUR-down.PNG) |



##### 等值面渲染结果

| 正视图                          | 侧视图                        | 俯视图                        |
| ------------------------------- | ----------------------------- | ----------------------------- |
| ![V-front](.\image\V-front.PNG) | ![V-side](.\image\V-side.PNG) | ![V-down](.\image\V-down.PNG) |



参考文献：

[ANDERS HAST Vis2014 Lecture2: Introduction to Python and VTK](http://www.cb.uu.se/~aht/Vis2014/lecture2.pdf)

课程PPT：  14.1 Demo_surface_volumerender 三维数据场渲染实现

素材：

[3d-nii-visualizer/sample_data/flair.nii.gz](https://github.com/adamkwolf/3d-nii-visualizer/blob/master/sample_data/flair.nii.gz)



3 请设计一个方法消除等值面渲染结果中的碎片化的面单元，如下图所示，并使用数据进行测试并且展示可视化结果。心脏CT图像：[image_lr.nii.gz](https://elearning.fudan.edu.cn/files/713949/download?wrap=1)



原理：使用低通滤波消除高频产生的图像混叠

```python
smooth = vtk.vtkImageGaussianSmooth()
smooth.SetDimensionality(3)
smooth.SetInputData(image)
smooth.SetStandardDeviations(1.75, 1.75, 0.0)
smooth.SetRadiusFactor(2)

subsampleSmoothed = vtk.vtkImageShrink3D()
subsampleSmoothed.SetInputConnection(smooth.GetOutputPort())
subsampleSmoothed.SetShrinkFactors(4, 4, 1)

isoSmoothed = vtk.vtkImageMarchingCubes()
isoSmoothed.SetInputConnection(smooth.GetOutputPort())
isoSmoothed.SetValue(0, 150)

stripper = vtk.vtkStripper() # 和之前相同
stripper.SetInputConnection(isoSmoothed.GetOutputPort())

```

结果

| before                          | after                         |
| ------------------------------- | ----------------------------- |
| ![Origin3](.\image\Origin3.PNG) | ![After3](.\image\After3.PNG) |

参考文献：

[VTKExamples/Python/ImageProcessing/IsoSubsample](https://lorensen.github.io/VTKExamples/site/Python/ImageProcessing/IsoSubsample/)