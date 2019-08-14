# You are at the top. Notice there are no bats hanging from the ceiling
# If there are weird bind errors like the mesh is not deforming correctly, compare 
#   the oct version of closest triangles to the one without oct


bl_info = {
    "name": "Surface Follow",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Extended Tools > Surface Follow",
    "description": "Doforms an object as the surface of another object changes",
    "warning": "You might have only imagined you were halucinating",
    "wiki_url": "",
    "category": '3D View'}

import bpy
import numpy as np
from numpy import newaxis as nax
np.seterr(all='ignore')
import bmesh
import time
from bpy.app.handlers import persistent


def apply_transforms(ob, co):
    """Get vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


def revert_transforms(ob, co):
    """Set world coords on object. 
    Run before setting coords to deal with object transforms
    if using apply_transforms()"""
    m = np.linalg.inv(ob.matrix_world)    
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


def set_key_coords(coords, key, ob):
    """Writes a flattened array to one of the object's shape keys."""
    ob.data.shape_keys.key_blocks[key].data.foreach_set("co", coords.ravel())
    ob.data.update()
    # Workaround for dependancy graph issue
    #ob.data.shape_keys.key_blocks[key].mute = True
    #ob.data.shape_keys.key_blocks[key].mute = False

    
def get_triangle_normals(tri_coords):
    '''does the same as get_triangle_normals 
    but I need to compare their speed'''    
    t0 = tri_coords[:, 0]
    t1 = tri_coords[:, 1]
    t2 = tri_coords[:, 2]
    return np.cross(t1 - t0, t2 - t0), t0


def get_bmesh(ob='empty'):
    '''Returns a bmesh. Works either in edit or object mode.
    ob can be either an object or a mesh.'''
    obm = bmesh.new()
    if ob == 'empty':
        mesh = bpy.context.object.data
    if 'data' in dir(ob):
        mesh = ob.data
        if ob.mode == 'OBJECT':
            obm.from_mesh(mesh)
        elif ob.mode == 'EDIT':
            obm = bmesh.from_edit_mesh(mesh)    
    else:
        mesh = ob
        obm.from_mesh(mesh)
    return obm


def set_coords(coords, ob='empty', use_proxy='empty'):
    """Writes a flattened array to the object. Second argument is for reseting
    offsets created by modifiers to avoid compounding of modifier effects"""
    if ob == 'empty':
        ob = bpy.context.object
    if use_proxy == 'empty':    
        ob.data.vertices.foreach_set("co", coords.ravel())
    else:        
        coords += use_proxy        
        ob.data.vertices.foreach_set("co", coords.ravel())
    ob.data.update()


def get_coords(ob='empty', proxy=False):
    '''Creates an N x 3 numpy array of vertex coords. If proxy is used the
    coords are taken from the object specified with modifiers evaluated.
    For the proxy argument put in the object: get_coords(ob, proxy_ob)'''
    if ob == 'empty':
        ob = bpy.context.object
    if proxy:
        dg = bpy.context.evaluated_depsgraph_get()
        mesh = ob.evaluated_get(dg).data
        
        #mesh = proxy.to_mesh(bpy.context.scene, True, 'PREVIEW')
        verts = mesh.vertices
    else:
        verts = ob.data.vertices
    v_count = len(verts)
    coords = np.zeros(v_count * 3)
    verts.foreach_get("co", coords)
    #if proxy:
        #bpy.data.meshes.remove(mesh)
    return coords.reshape(v_count, 3)


def triangulate(ob='empty', proxy=False):
    '''Requires a mesh. Returns an index array for viewing
    the coordinates as triangles. Store this!!! rather than recalculating
    every time. !!!Could use for_each_get with the mesh and polygons if
    all the faces have 3 points!!! Could also write bmesh to mesh and use
    foreach_get'''    
    if ob == 'empty':
        ob = bpy.context.object
    if proxy:
        mods = True
    else:
        mods = False
    #proxy = ob.to_mesh(bpy.context.scene, mods, 'PREVIEW')
    dg = bpy.context.evaluated_depsgraph_get()
    proxy = ob.evaluated_get(dg).data    
    
    obm = get_bmesh(proxy)        
    bmesh.ops.triangulate(obm, faces=obm.faces)
    obm.to_mesh(proxy)        
    count = len(proxy.polygons)    
    tri_idx = np.zeros(count * 3, dtype=np.int64)        
    proxy.polygons.foreach_get('vertices', tri_idx)        
    #bpy.data.meshes.remove(proxy)
    obm.free()
    return tri_idx.reshape(count, 3)


def barycentric_generate(hits, tris):
    '''Create scalars to be used by points and triangles'''
    # where the hit lands on the two tri vecs
    tv = tris[:, 1] - tris[:, 0]
    hv = hits - tris[:, 0]
    d1a = np.einsum('ij, ij->i', hv, tv)
    d1b = np.einsum('ij, ij->i', tv, tv)
    scalar1 = np.nan_to_num(d1a / d1b)

    t2v = tris[:, 2] - tris[:, 0]
    d2a = np.einsum('ij, ij->i', hv, t2v)
    d2b = np.einsum('ij, ij->i', t2v, t2v)
    scalar2 = np.nan_to_num(d2a / d2b)
    
    # closest point on edge segment between the two points created above
    cp1 = tv * scalar1[:, nax]
    cp2 = t2v * scalar2[:, nax]
    cpvec = cp2 - cp1
    cp1_at = tris[:,0] + cp1
    hcp = hits - cp1_at # this is cp3 above. Not sure what's it's for yet
    dhcp = np.einsum('ij, ij->i', hcp, cpvec)
    d3b = np.einsum('ij, ij->i', cpvec, cpvec)
    hcp_scalar = np.nan_to_num(dhcp / d3b)
    hcp_vec = cpvec * hcp_scalar[:, nax]    
    
    # base of tri on edge between first two points
    d3 = np.einsum('ij, ij->i', -cp1, cpvec)
    scalar3 = np.nan_to_num(d3 / d3b)
    base_cp_vec = cpvec * scalar3[:, nax]
    base_on_span = cp1_at + base_cp_vec

    # Where the point occurs on the edge between the base of the triangle
    #   and the cpoe of the base of the triangle on the cpvec    
    base_vec = base_on_span - tris[:,0]
    dba = np.einsum('ij, ij->i', hv, base_vec)
    dbb = np.einsum('ij, ij->i', base_vec, base_vec)
    scalar_final = np.nan_to_num(dba / dbb)
    p_on_bv = base_vec * scalar_final[:, nax]
    perp = (p_on_bv) - (cp1 + base_cp_vec)
    return scalar1, scalar2, hcp_scalar, scalar3, scalar_final


def barycentric_remap_multi(tris, sc1, sc2, sc3, sc4, sc5, scale):
    '''Uses the scalars generated by barycentric_generate() to remap the points
    on the triangles and off the surface as the surface mesh changes'''
    # where the hit lands on the two tri vecs
    tv = tris[:, 1] - tris[:, 0]
    t2v = tris[:, 2] - tris[:, 0]
    
    # closest point on edge segment between the two points created above
    cp1 = tv * sc1[:, nax]
    cp2 = t2v * sc2[:, nax]
    cpvec = cp2 - cp1
    cp1_at = tris[:,0] + cp1
    hcp_vec = cpvec * sc3[:, nax]    
    
    # base of tri on edge between first two points
    base_cp_vec = cpvec * sc4[:, nax]
    base_on_span = cp1_at + base_cp_vec

    # Where the point occurs on the edge between the base of the triangle
    #   and the cpoe of the base of the triangle on the cpvec    
    base_vec = base_on_span - tris[:,0]
    p_on_bv = base_vec * sc5[:, nax]
    perp = (p_on_bv) - (cp1 + base_cp_vec)

    # get the average length of the two vectors and apply it to the cross product
    cross = np.cross(tv, t2v)
    sq = np.sqrt(np.einsum('ij,ij->i', cross, cross))
    x1 = np.einsum('ij,ij->i', tv, tv)
    x2 = np.einsum('ij,ij->i', t2v, t2v)
    av_root = np.sqrt((x1 + x2) / 2)
    cr_root = (cross / sq[:, nax]) * (av_root * scale)[:, nax]    

    return tris[:,0] + cp1 + hcp_vec + perp + cr_root


def project_points(points, tri_coords):
    '''Using this to get the points off the surface
    Takes the average length of two vecs off triangles
    and applies it to the length of the normals.
    This way the normal scales with the mesh and with
    changes to the individual triangle vectors'''
    t0 = tri_coords[:, 0]
    t1 = tri_coords[:, 1]
    t2 = tri_coords[:, 2]
    tv1 = t1 - t0
    tv2 = t2 - t0
    cross = np.cross(tv1, tv2)
    
    # get the average length of the two vectors and apply it to the cross product
    sq = np.sqrt(np.einsum('ij,ij->i', cross, cross))
    x1 = np.einsum('ij,ij->i', tv1, tv1)
    x2 = np.einsum('ij,ij->i', tv2, tv2)
    av_root = np.sqrt((x1 + x2) / 2)
    cr_root = (cross / sq[:, nax]) * av_root[:, nax]    
     
    v1 = points - t0
    v1_dots = np.einsum('ij,ij->i', cr_root, v1)
    n_dots = np.einsum('ij,ij->i', cr_root, cr_root)
    scale = np.nan_to_num(v1_dots / n_dots)
    offset = cr_root * scale[:, nax]
    drop = points - offset # The drop is used by the barycentric generator as points in the triangles
    return drop, scale


def nearest_triangles(surface_coords, follower_coords, tris): 
    '''Basic N-squared method for getting triangls.
    Slow on large sets. Using octree instead'''                   # Before there were octrees...
    follow_co = follower_coords.astype(np.float32)                # There were huge sets of tiles.
    surface_co = surface_coords.astype(np.float32)                # Before the dark times. Before the empire.  
    means = np.mean(surface_co[tris], axis=1)
    difs = follow_co[:, nax] - means
    dots = np.einsum('ijk, ijk->ij', difs, difs)
    sorts = np.argmin(dots, axis=1)    
    return sorts    

        
def nearest_triangles_oct(surface_coords, follower_coords, tris):  # octree
    '''Use octree to find nearest triangles centers'''
    # yes I really created an octree inline. What's the world coming to... I know, I know
    follow_co = follower_coords.astype(np.float32)
    surface_co = surface_coords.astype(np.float32)
    fill_me = np.zeros(len(follow_co), dtype=np.int)
    
    means = np.mean(surface_co[tris], axis=1)
    # 2: Get the mean of the surface tri means.
    box_mean = np.mean(means, axis=0)
    #bpy.data.objects['Empty'].location = box_mean
    # 3: Make 8 boxes:
    s1 = means < box_mean[nax]
    b0 = np.all(s1, axis=1)
    m0 = np.mean(means[b0], axis=0)    
    
    s1[:,0] = ~s1[:,0]
    b1 = np.all(s1, axis=1)
    m1 = np.mean(means[b1], axis=0)
    
    s1[:,1] = ~s1[:,1]
    b2 = np.all(s1, axis=1)
    m2 = np.mean(means[b2], axis=0)

    s1[:,0] = ~s1[:,0]
    b3 = np.all(s1, axis=1)
    m3 = np.mean(means[b3], axis=0) 

    s1[:,2] = ~s1[:,2]
    b4 = np.all(s1, axis=1)
    m4 = np.mean(means[b4], axis=0)

    s1[:,0] = ~s1[:,0]
    b5 = np.all(s1, axis=1)
    m5 = np.mean(means[b5], axis=0)
    
    s1[:,1] = ~s1[:,1]
    b6 = np.all(s1, axis=1)
    m6 = np.mean(means[b6], axis=0)

    s1[:,0] = ~s1[:,0]
    b7 = np.all(s1, axis=1)
    m7 = np.mean(means[b7], axis=0)

    m_list = np.array([m0, m1, m2, m3, m4, m5, m6, m7])
    
    m_b_dif = m_list - box_mean
    mean_mags = np.sqrt(np.einsum('ij,ij->i', m_b_dif, m_b_dif)) * 1.1# between the mean box and the means
    
    # Here we convert the octree into eightballs to eliminate special case distance errors. ?octballs?
    inny = []
    eight = np.arange(8)
    for i in eight:
        dif = means - m_list[i]
        dist = np.sqrt(np.einsum('ij,ij->i', dif, dif))
        in_range = dist < mean_mags[i]    
        inny.append(in_range)
    
    # For reference the above code can be done without iterating (it's about the same speed if not slower)
    #dif = np.expand_dims(means, axis=1) - m_list
    #d = np.sqrt(np.einsum('ijk, ijk->ij', dif, dif))
    #inny = (d < mean_mags).T
    
    b_list = np.array(inny) # first cull step, eliminate boxes with no tris. 
    b_pos = np.any(b_list, axis=1) # first cull step, eliminate boxes with no tris. 

    box_set = b_list[b_pos] # bool of only the sets of triangles in boxes
    mean_set = m_list[b_pos] 
    
    m_f_dif = follow_co[:, nax] - mean_set
    m_f_d = np.einsum('ijk, ijk->ij', m_f_dif, m_f_dif)
    m_f_min = np.argmin(m_f_d, axis=1) # which box is the closest to each vert
    
    tri_indexer = np.arange(len(means))
    for i in range(len(box_set)):
        dif = follow_co[m_f_min == i][:, nax] - means[box_set[i]]
        dif_d = np.einsum('ijk, ijk->ij', dif, dif)
        amin = np.argmin(dif_d, axis=1) 
        trises = tri_indexer[box_set[i]][amin] # ! ding ding ding, now need the verts in the follow
        vertses = m_f_min == i
        fill_me[vertses] = trises

    return fill_me


def multi_bind():
    x = 5
    obj = bpy.context.object 
    if obj == None:
        return -1
    list = [i for i in bpy.context.selected_objects if i.type == 'MESH']
    count = len(list)
    # sort active object and cull objects that are not meshes:
    if count < 2:
        return -1
    di = bpy.context.scene.surface_follow_data_set            
    di['surfaces'][obj.name] = obj
    di_followers = di['objects']
    for i in bpy.context.selected_objects:
        if (i.type == 'MESH') & (i != obj): 
            if i.data.shape_keys == None:
                i.shape_key_add(name='Basis')    
            if 'surface follow' not in i.data.shape_keys.key_blocks:
                i.shape_key_add(name='surface follow')        
                i.data.shape_keys.key_blocks['surface follow'].value=1
            #a = transform_matrix(get_coords(obj, obj), obj)
            a = apply_transforms(obj, get_coords(obj, obj))
            
            #b = transform_matrix(get_coords(i), i)    
            b = apply_transforms(i, get_coords(i))    
            tris = triangulate(obj, proxy=True)
            #reg = nearest_triangles(a, b, tris)
            oct = nearest_triangles_oct(a, b, tris)
            tri_indexer = tris[oct]
            tri_coords = a[tri_indexer]
            hits, length = project_points(b, tri_coords)
            scalars = barycentric_generate(hits, a[tri_indexer])
            
            # Create dictionary items:
            di_followers[i.name] = {}
            di_followers[i.name]['surface'] = obj
            di_followers[i.name]['tri_indexer'] = tri_indexer
            di_followers[i.name]['scalars'] = scalars
            di_followers[i.name]['length'] = length
            di_followers[i.name]['surface_coords'] = a


def multi_update():
    obs = bpy.data.objects
    di = bpy.context.scene.surface_follow_data_set
    s_coords = {}
    
    # if an object no longer has valid data it goes into a list to be deleted when done iterating 
    cull_list = [] 
    
    for i in di['surfaces']:
        try:
            #s_coords[i] = transform_matrix(get_coords(obs[i], obs[i]), obs[i])
            s_coords[i] = apply_transforms(obs[i], get_coords(obs[i], obs[i]))
        except (KeyError, RuntimeError, ReferenceError):
            cull_list.append(i)

    #for i in di['objects']:
    for i, value in di['objects'].items():
        try:            
            child = obs[i]
            coords = s_coords[value['surface'].name]
            project = barycentric_remap_multi(coords[value['tri_indexer']], value['scalars'][0], value['scalars'][1], value['scalars'][2], value['scalars'][3], value['scalars'][4], value['length'])
            set_key_coords(revert_transforms(child, project), 'surface follow', child)
        except (KeyError, RuntimeError, ReferenceError):
            cull_list.append(i)

    for i in cull_list:
        del i


def test_thingy():
    print('doing something every frame (like bathing or possibly eating a mountain goat)')


#@persistent
def SF_handler(scene, override=False):
    multi_update()


def SF_timer():
    multi_update()
    return 0
    

if bpy.app.timers.is_registered(SF_timer):
    bpy.app.timers.unregister(SF_timer)

        
def remove_handler(type):
    '''Deletes handler from the scene'''
    if type == 'scene':
        if bpy.app.timers.is_registered(SF_timer):
            bpy.app.timers.unregister(SF_timer)
    if type == 'frame':
        if SF_handler in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.remove(SF_handler)


def add_handler(type):
    '''adds handler from the scene'''
    if type == 'scene':        
        bpy.app.timers.register(SF_timer)
        print("added the timer ---------")
    if type == 'frame':
        bpy.app.handlers.frame_change_post.append(SF_handler)

    
# run on prop callback
def toggle_display(self, context):
    if bpy.context.scene.surface_follow_on:
        add_handler('scene')
        remove_handler('frame')
        bpy.context.scene['surface_follow_frame'] = False
        
    elif bpy.context.scene.surface_follow_frame:
        add_handler('frame')
        remove_handler('scene')    
        bpy.context.scene['surface_follow_on'] = False
    else:
        remove_handler('scene')
        remove_handler('frame')


# Properties-----------------------------------:
def create_properties():            

    bpy.types.Scene.surface_follow_on = bpy.props.BoolProperty(name="Scene Update", 
        description="For toggling the dynamic tension map", 
        default=False, update=toggle_display)

    bpy.types.Scene.surface_follow_frame = bpy.props.BoolProperty(name="Frame Update", 
        description="For toggling the dynamic tension map", 
        default=False, update=toggle_display)

    bpy.types.Scene.surface_follow_data_set = {} 
    bpy.types.Scene.surface_follow_data_set['surfaces'] = {}
    bpy.types.Scene.surface_follow_data_set['objects'] = {}

    
def remove_properties():            
    '''Walks down the street and gets me a coffee'''
    del(bpy.types.Scene.surface_follow_on)
    del(bpy.types.Scene.surface_follow_frame)
    del(bpy.types.Scene.surface_follow_data_set)


# Create Classes-------------------------------:
class BindToSurface(bpy.types.Operator):
    '''Bind To Surface'''
    bl_idname = "scene.bind_to_surface"
    bl_label = "bind to surface"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        x = multi_bind()
        if x == -1:    
            self.report({'ERROR'}, 'Select at least two objects')
        return {'FINISHED'}


class ToggleSurfaceFollow(bpy.types.Operator):
    '''Toggle Surface Follow Update'''
    bl_idname = "scene.toggle_surface_follow"
    bl_label = "surface follow updater"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        toggle_display()
        return {'FINISHED'}


class UpdateOnce(bpy.types.Operator):
    '''Surface Update'''
    bl_idname = "scene.surface_update_once"
    bl_label = "update surface one time"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        SF_handler(None, True)
        #SF_handler()
        return {'FINISHED'}


class Patreon(bpy.types.Operator):
    """Loads my patreon page"""
    bl_idname = "scene.my_patreon"
    bl_label = "Wrap Tools Patreon"
    def execute(self, context):
        import webbrowser
        webbrowser.open("https://www.patreon.com/The_3D_Advantage")
        self.report({'INFO'}, 'Patreon, The3dAdvantage@gmail.com')
        return {'FINISHED'}


class Paypal(bpy.types.Operator):
    """Loads my paypal donate button"""
    bl_idname = "scene.my_paypal"
    bl_label = "Wrap Tools Paypal"
    def execute(self, context):
        import webbrowser
        webbrowser.open("https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=4T4WNFQXGS99A")
        self.report({'INFO'}, 'Paypal, The3dAdvantage@gmail.com')
        return {'FINISHED'}


class SurfaceFollowPanel(bpy.types.Panel):
    """Surface Follow Panel"""
    bl_label = "Surface Follow Panel"
    bl_idname = "SURFACE_FOLLOW_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"
    #gt_show = True
    
    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Surface Follow")
        col.operator("scene.bind_to_surface", text="Bind to Surface")
        col.operator("scene.surface_update_once", text="Update Once", icon='FILE_REFRESH')        
        if not bpy.context.scene.surface_follow_frame:    
            col.prop(bpy.context.scene ,"surface_follow_on", text="Scene Update", icon='SCENE_DATA')               
        if not bpy.context.scene.surface_follow_on:            
            col.prop(bpy.context.scene ,"surface_follow_frame", text="Frame Update", icon='PLAY')               
        col.label(text="Support/Guilt Trip")
        col.operator("scene.my_patreon", text="Patreon", icon='FUND')        
        col.operator("scene.my_paypal", text="Donate", icon='FUND')        

# Register Clases -------------->>>
def register():
    create_properties()
    bpy.utils.register_class(SurfaceFollowPanel)
    bpy.utils.register_class(BindToSurface)
    bpy.utils.register_class(UpdateOnce)
    bpy.utils.register_class(ToggleSurfaceFollow)
    bpy.utils.register_class(Patreon)
    bpy.utils.register_class(Paypal)


def unregister():
    remove_properties()
    bpy.utils.unregister_class(BindToSurface)
    bpy.utils.unregister_class(UpdateOnce)
    bpy.utils.unregister_class(ToggleSurfaceFollow)
    bpy.utils.unregister_class(SurfaceFollowPanel)
    bpy.utils.unregister_class(Patreon)
    bpy.utils.unregister_class(Paypal)


if __name__ == "__main__":
    register()
