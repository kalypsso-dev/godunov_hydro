# trace generated using paraview version 5.12.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XDMF Reader'
test_sod_2d_mainxmf = XDMFReader(registrationName='test_sod_2d_main.xmf', FileNames=['@CMAKE_CURRENT_BINARY_DIR@/test_sod_2d_main.xmf'])

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on test_sod_2d_mainxmf
test_sod_2d_mainxmf.CellArrayStatus = ['e_tot', 'level', 'rho', 'rho_vx', 'rho_vy']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
test_sod_2d_mainxmfDisplay = Show(test_sod_2d_mainxmf, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
test_sod_2d_mainxmfDisplay.Representation = 'Surface'

# reset view to fit data
#renderView1.ResetCamera(False, 0.9)
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5, 0.015625, 3.35]
renderView1.CameraFocalPoint = [0.5, 0.015625, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
test_sod_2d_mainxmfDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'level'
levelLUT = GetColorTransferFunction('level')

# get opacity transfer function/opacity map for 'level'
levelPWF = GetOpacityTransferFunction('level')

# get 2D transfer function for 'level'
levelTF2D = GetTransferFunction2D('level')

animationScene1.GoToLast()

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=test_sod_2d_mainxmf)

# set active source
SetActiveSource(test_sod_2d_mainxmf)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=slice1.SliceType)

# destroy slice1
Delete(slice1)
del slice1

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=test_sod_2d_mainxmf)

# Properties modified on plotOverLine1
plotOverLine1.Point1 = [0.0, 0.015625, 0.0]
plotOverLine1.Point2 = [1.0, 0.015625, 0.0]

# show data in view
plotOverLine1Display = Show(plotOverLine1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
plotOverLine1Display.Representation = 'Surface'

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')

# show data in view
plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, 'XYChartRepresentation')

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

# Properties modified on plotOverLine1Display_1
plotOverLine1Display_1.SeriesOpacity = ['arc_length', '1', 'e_tot', '1', 'level', '1', 'rho', '1', 'rho_vx', '1', 'rho_vy', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display_1.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'e_tot', '0', 'level', '0', 'rho', '0', 'rho_vx', '0', 'rho_vy', '0', 'vtkValidPointMask', '0']
plotOverLine1Display_1.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'arc_length', '1', 'e_tot', '1', 'level', '1', 'rho', '1', 'rho_vx', '1', 'rho_vy', '1', 'vtkValidPointMask', '1']
plotOverLine1Display_1.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'arc_length', '2', 'e_tot', '2', 'level', '2', 'rho', '2', 'rho_vx', '2', 'rho_vy', '2', 'vtkValidPointMask', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'e_tot', '0', 'level', '0', 'rho', '0', 'rho_vx', '0', 'rho_vy', '0', 'vtkValidPointMask', '0']
plotOverLine1Display_1.SeriesMarkerSize = ['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'arc_length', '4', 'e_tot', '4', 'level', '4', 'rho', '4', 'rho_vx', '4', 'rho_vy', '4', 'vtkValidPointMask', '4']

# save data
SaveData('@CMAKE_CURRENT_BINARY_DIR@/sod_numerical_solution.csv', proxy=plotOverLine1, PointDataArrays=['arc_length', 'e_tot', 'level', 'rho', 'rho_vx', 'rho_vy', 'vtkValidPointMask'])

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1358, 796)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5, 0.015625, 3.35]
renderView1.CameraFocalPoint = [0.5, 0.015625, 0.0]
renderView1.CameraParallelScale = 0.5002440810494413


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, useful when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple split view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------
