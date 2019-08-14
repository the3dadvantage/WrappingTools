#----------------------------------------------------------
# File __init__.py
#----------------------------------------------------------
 
# If you are reading this you are a rare person indeed.
# Perhaps you need a little more time on the grill
 
# Addon info
bl_info = {
    "name": "Wrapping Tools",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Extended Tools",
    "description": "Tools for deforming around complex surfaces",
    "warning": "If God loses self-confidence He will be an atheist",
    "wiki_url": "",
    "category": '3D View'}

if "bpy" in locals():
    import imp
    imp.reload(SurfaceFollow28)
    imp.reload(UVShape28)
    print("Reloaded UV Shape and Surface Follow")
else:
    from . import SurfaceFollow28, UVShape28
    print("Imported Surface Follow and UV Shape")

   
def register():
    SurfaceFollow28.register()
    UVShape28.register()

    
def unregister():
    SurfaceFollow28.unregister()
    UVShape28.unregister()

    
if __name__ == "__main__":
    register()
