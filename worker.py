# worker.py (Improved)
from abaqus import *
from abaqusConstants import *
import sys
import os

# --- 1. SETUP ---
radius = 10.0
load_mag = 100.0
args = sys.argv
if len(args) > 1:
    try:
        radius = float(args[-2])
        load_mag = float(args[-1])
    except:
        pass

model_name = 'Model-1'
job_name = 'Sim_R' + str(int(radius)) + '_L' + str(int(load_mag))

Mdb()
my_model = mdb.models[model_name]

# --- 2. GEOMETRY (Standard) ---
s = my_model.ConstrainedSketch(name='__profile__', sheetSize=200.0)
s.rectangle(point1=(0.0, 0.0), point2=(100.0, 50.0))
s.CircleByCenterPerimeter(center=(50.0, 25.0), point1=(50.0 + radius, 25.0))

p = my_model.Part(name='Plate', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
p.BaseShell(sketch=s)

# --- 3. PHYSICS ---
my_mat = my_model.Material(name='Steel')
my_mat.Elastic(table=((210000.0, 0.3), ))
my_section = my_model.HomogeneousSolidSection(name='Section-1', material='Steel', thickness=1.0)
region = p.Set(faces=p.faces, name='Set-1')
p.SectionAssignment(region=region, sectionName='Section-1')

a = my_model.rootAssembly
a.Instance(name='Plate-1', part=p, dependent=ON)

my_model.StaticStep(name='ApplyLoad', previous='Initial')

# BCs
left_edge = a.instances['Plate-1'].edges.findAt(((0.0, 25.0, 0.0), ))
region_bc = a.Set(edges=left_edge, name='LeftEdge')
my_model.EncastreBC(name='FixLeft', createStepName='Initial', region=region_bc)

# Loads
right_edge = a.instances['Plate-1'].edges.findAt(((100.0, 25.0, 0.0), ))
region_load = a.Surface(side1Edges=right_edge, name='RightEdge')
my_model.Pressure(name='Pull', createStepName='ApplyLoad', region=region_load, magnitude=-load_mag)

# --- 4. MESH (Refined slightly for stability) ---
# Use size 4.0 (a bit finer) to handle bigger holes better
# Note: Keep an eye on node limit, but 4.0 should be safe (~300 nodes)
p.seedPart(size=4.0, deviationFactor=0.1, minSizeFactor=0.1)
p.generateMesh()

# --- 5. RUN ---
my_job = mdb.Job(name=job_name, model=model_name)
my_job.submit()
my_job.waitForCompletion()

# --- 6. SAFETY DATA EXTRACTION ---
from odbAccess import openOdb

try:
    # Try to open results
    odb = openOdb(path=job_name + '.odb')
    
    # distinct check: did the step actually finish?
    if 'ApplyLoad' in odb.steps:
        last_frame = odb.steps['ApplyLoad'].frames[-1]
        stress_field = last_frame.fieldOutputs['S']
        
        max_stress = 0.0
        for val in stress_field.values:
            if val.mises > max_stress:
                max_stress = val.mises
        
        # SUCCESS: Write data
        with open('data_log.txt', 'a') as f:
            f.write(str(radius) + "," + str(load_mag) + "," + str(max_stress) + "\n")
            
    else:
        # Simulation ran but crashed mid-way
        with open('error_log.txt', 'a') as f:
            f.write("Crash (Step missing): " + job_name + "\n")

    odb.close()

except Exception as e:
    # Total failure (file didn't open)
    with open('error_log.txt', 'a') as f:
        f.write("Crash (File Error): " + str(e) + "\n")