import nibabel as nib
import vtk

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


# to scalars
image.AllocateScalars(10, 1)
for z in range(dims[2]):
	for y in range(dims[1]):
		for x in range(dims[0]):
			scalardata = image_data[x][y][z]
			image.SetScalarComponentFromDouble(x, y , z, 0, scalardata)

# filter
Extractor = vtk.vtkMarchingCubes()
Extractor.SetInputData(image)
Extractor.SetValue(0, 100)
stripper = vtk.vtkStripper()
stripper.SetInputConnection(Extractor.GetOutputPort())


# Mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(stripper.GetOutputPort())

# Actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
colors = vtk.vtkNamedColors()
actor.GetProperty().SetColor(colors.GetColor3d("Wheat"))
mapper.ScalarVisibilityOff()

# render
renderer = vtk.vtkRenderer()
renderer.SetBackground(colors.GetColor3d("SlateGray"))
renderer.AddActor(actor)

#render window
render_window = vtk.vtkRenderWindow()
render_window.SetWindowName("Brain")
render_window.SetSize(400, 400)
render_window.AddRenderer(renderer)

# interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor.Initialize()
render_window.Render()
interactor.Start()


