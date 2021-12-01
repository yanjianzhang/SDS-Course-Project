import nibabel as nib
import vtk

# 1. 读取数据

Path = r"image_lr.nii.gz"

image = nib.load(Path)
image_data = image.get_fdata()
dims = image.shape
spacing = (image.header['pixdim'][1], image.header['pixdim'][2], image.header['pixdim'][3])

image = vtk.vtkImageData()
image.SetDimensions(dims[0], dims[1], dims[2])
image.SetSpacing(spacing[0], spacing[1], spacing[2])
image.SetOrigin(0, 0, 0)


def NumberOfTriangles(pd):
	"""
	Count the number of triangles.
	:param pd: vtkPolyData.
	:return: The number of triangles.
	"""
	cells = pd.GetPolys()
	numOfTriangles = 0
	idList = vtk.vtkIdList()
	for i in range(0, cells.GetNumberOfCells()):
		cells.GetNextCell(idList)
		# If a cell has three points it is a triangle.
		if idList.GetNumberOfIds() == 3:
			numOfTriangles += 1
	return numOfTriangles

# to scalars
image.AllocateScalars(10, 1)
for z in range(dims[2]):
	for y in range(dims[1]):
		for x in range(dims[0]):
			scalardata = image_data[x][y][z]
			image.SetScalarComponentFromDouble(x, y , z, 0, scalardata)


# filter
# Extractor = vtk.vtkMarchingCubes()
# Extractor.SetInputData(image)
# Extractor.SetValue(0, 100)

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
#
# connect = vtk.vtkPolyDataConnectivityFilter()
# connect.SetInputConnection(Extractor.GetOutputPort())
# connect.SetExtractionModeToLargestRegion()

stripper = vtk.vtkStripper()
stripper.SetInputConnection(isoSmoothed.GetOutputPort())




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
# renderer.SetAmbient(0,0,0)
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


