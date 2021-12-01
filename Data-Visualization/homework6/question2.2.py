import nibabel as nib
import vtk
import numpy as np

# 1. 读取数据

Path = r"flair.nii.gz"

image = nib.load(Path)
image_data = image.get_fdata()
dims = image.shape
spacing = (image.header['pixdim'][1], image.header['pixdim'][2], image.header['pixdim'][3])

image = vtk.vtkImageData()
image.SetDimensions(dims[0], dims[1], dims[2])
image.SetSpacing(spacing[0], spacing[1], spacing[2])
image.SetOrigin(0, 0, 0)
image.SetExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)

intRange = (20, 500)

max_u_shot  = 128

# to scalars
image.AllocateScalars(10, 1)
for z in range(dims[2]):
	for y in range(dims[1]):
		for x in range(dims[0]):
			scalardata = image_data[x][y][z]
			scalardata = max(min(intRange[1], scalardata ), intRange[0])
			scalardata = max_u_shot*np.float(scalardata- intRange[0])/np.float(intRange[1] - intRange[0])
			image.SetScalarComponentFromDouble(x, y , z, 0, scalardata)


# filter
opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(0, 0.0)
opacityTransferFunction.AddSegment(23, 0.3, 128, 0.5)
opacityTransferFunction.ClampingOff()

colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.AddRGBSegment(0, 0.0, 0.0, 0.0, 20, 0.2, 0.2, 0.2)
colorTransferFunction.AddRGBSegment(20, 0.1, 0.1, 0, 128, 1, 1, 0)

gradientTransferFunction = vtk.vtkPiecewiseFunction()
gradientTransferFunction.AddPoint(0,0.0)
gradientTransferFunction.AddSegment(100, 0.1, 1000, 0.3)



volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetGradientOpacity(gradientTransferFunction)
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.SetAmbient(1)
volumeProperty.SetDiffuse(0.9)
volumeProperty.SetSpecular(0.8)
volumeProperty.SetSpecularPower(10)

# mapper
# compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
# mapper = vtk.vtkVolumeRayCastMapper()
# mapper.SetVolumeRayCastFunction(compositeFunction)
# mapper.SetInputData(image)
# mapper.SetImageSampleDistance(5.0)

# colorFunc = vtk.vtkColorTransferFunction()
# colorFunc.AddRGBPoint(50, 1.0, 0.0, 0.0)
# colorFunc.AddRGBPoint(100, 0.0, 1.0, 0.0)
# colorFunc.AddRGBPoint(150, 0.0, 0.0, 1.0)

# The previous two classes stored properties.
#  Because we want to apply these properties to the volume we want to render,
# we have to store them in a class that stores volume properties.
# volumeProperty = vtk.vtkVolumeProperty()
# volumeProperty.SetColor(colorFunc)

mapper = vtk.vtkFixedPointVolumeRayCastMapper()

mapper.SetInputData(image)
mapper.SetImageSampleDistance(5.0)


# Actor
volume = vtk.vtkVolume()
volume.SetMapper(mapper)
volume.SetProperty(volumeProperty)
colors = vtk.vtkNamedColors()

# render
renderer = vtk.vtkRenderer()
renderer.SetBackground(colors.GetColor3d("SlateGray"))
renderer.AddVolume(volume)

#render window
render_window = vtk.vtkRenderWindow()
render_window.SetWindowName("Brain")
render_window.SetSize(400, 400)
render_window.AddRenderer(renderer)

light = vtk.vtkLight()
light.SetColor(1,1,1)
renderer.AddLight(light)

# interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor.Initialize()
render_window.Render()
interactor.Start()

# gradientFilter = vtk.vtkImageGradient()
# gradientFilter.SetInputData(image)
# gradientFilter.SetDimensionality(3)
# gradientFilter.Update()
# gradimage = gradientFilter.GetOutputDataObject(0)
#
# magnitude = vtk.vtkImageMagnitude()
# magnitude.SetInputConnection(gradientFilter.GetOutputPort())
# magnitude.Update()
# imageCast = vtk.vtkImageCast()
# imageCast.SetInputConnection(magnitude.GetOutputPort())
#
#
