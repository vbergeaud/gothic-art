# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 17:14:04 2020

@author: VB144235
"""
import math
from shapely.geometry import Point,Polygon,LineString
from shapely.affinity import affine_transform,rotate,translate
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import copy

def rectangle(w,h):
    rect=Polygon([[-w*0.5,h*0.5],[w*0.5,h*0.5],[w*0.5,-h*0.5],[-w*0.5,-h*0.5]])
    return rect

def circle(P,R):
    return P.buffer(R)

def translate(shape,dx,dy):
    mat=[1,0,0,1,dx,dy]
    return affine_transform(shape,mat)

def broken_arc(w,h,ch,t):
    alpha=2*math.atan(w/h*0.5)
    r=h/math.sin(alpha)
    c1=Point(w/2-r,0).buffer(r)
    c2=Point(r-w/2,0).buffer(r)
    inter=c1.intersection(c2)
    cutrect=affine_transform(rectangle(2*w,2*h+ch),[1,0,0,1,0,-2*ch])
    rect=affine_transform(rectangle(w,ch),[1,0,0,1,0,-ch*0.5])
    arc=inter.union(rect)
    arc=arc.difference(cutrect)
    arc=arc.exterior.buffer(t)
    return arc

""" 
shape : shape to be rotated
n : number of copies (including original shape)
copies the shape and rotates by i/k*2*pi radians
returns list of shapes
"""
def multi_rotate(shape,n):
    u=[shape]
    for i in range(n-1):
        alpha=360/n
        shape=rotate(shape,alpha,(0.0,0.0))
        u.append(shape)
    return u
    
"""
coffin shape 
Rint : inner radius
Rmid : radius of the outer part of the coffin body
alpha : angle of the body in degrees 
"""

def new_coffin(Rint, Rmid, alpha):
    alpha_rad = alpha/180.0*math.pi
    cosa2 = math.cos(alpha_rad*.5)
    sina2 = math.sin(alpha_rad*.5)
    tana2 = math.sin(alpha_rad*.5)
    O = Point(0.0,0.0)
    Cint = circle(O,Rint)
    Cmid = circle(O,Rmid)
    A = Point(Rmid*(cosa2+2*tana2*sina2),-Rmid*sina2)
    B = Point(Rmid*(cosa2+2*tana2*sina2),Rmid*sina2)
    C = Point(Rmid*cosa2,-Rmid*sina2)
    D = Point(Rmid*cosa2,Rmid*sina2)
    Cbot = circle(A,A.distance(D))
    Ctop = circle(B,B.distance(C))
    triangle = Polygon([C,D,O])
    shape = triangle.difference(Cint)
    shape2 = Cbot.intersection(Ctop)
    shape_out = shape.union(shape2)
    r = rosace(Rmid*sina2*0.5,3)
    r = translate(r, Rmid*(cosa2+tana2*sina2),0)
    shape_in = r.union(shape)
   
    return shape_out,shape_in

def coffin(Rint,wmid,Rmid,lmid,wtop,htop,Rtop):
    ltot = lmid+Rint
    L = math.sqrt((ltot-Rmid)*(ltot-Rmid)-Rmid*Rmid)
    alpha = math.atan(Rmid/L)
    tri = Polygon([[0,0],[math.cos(alpha)*L, -math.sin(alpha)*L],[math.cos(alpha)*L, math.sin(alpha)*L]])
    c = circle(lmid+Rint-Rmid,0,Rmid)
    union = tri.union(c)
    trans = 2*Rmid-wmid
    union2 = translate(union,0,-trans/2)
    union3 = translate(union, 0,trans/2)
    inter = union3.intersection(union2)
    cint = circle(O,Rint)
    inter = inter.difference(cint)    
    c1 = circle(ltot,0,Rtop)
    c2 = translate(c1,0,-Rtop+wtop/2)
    c3 = translate(c1,0,Rtop-wtop/2)
    eye = c2.intersection(c3)
    return eye.union(inter)

def double_coffin(Rint,wmid,Rmid,lmid,wtop,htop,Rtop):
    wmid_single = wmid/1.8
    c = coffin(Rint,wmid_single,Rmid,lmid,wtop,htop,Rtop)
    rot_angle = math.atan(wmid*0.5)/(Rint+lmid)*360/math.pi*2
    c2=rotate(c,rot_angle,(0,0))
    return rotate(c.union(c2),-rot_angle*0.5,(0,0))
    
def rosace (r,n):
    circle = Point(r*0.9,0).buffer(r)
    shape = circle
    for iangle in range(n):
        angle = iangle*2*math.pi/n
        circle_bis = rotate(circle,angle*360/2/math.pi,(0,0))
        shape = shape.union(circle_bis)
    return shape

"""
Rint : radius of the centers of the outer circles
Rext : radius of the outer circles (total radius will be Rint+Rext)
N : number of outer circles
inner_circle_reduction_factor : multiplication factor for the radius of the inner circle (applied to Rint)
surrounded : creates an outer circle if True
"""
def new_rosace(Rint,Rext,N,inner_circle_reduction_factor = 0.7, surrounded=False):
    circle = Point(0,0).buffer(Rint*inner_circle_reduction_factor)
    shape= circle
    for i in range(N):
        cosa = math.cos(i*2*math.pi/N)
        sina = math.sin(i*2*math.pi/N)
        circle2 = Point (Rint*cosa,Rint*sina).buffer(Rext)
        shape=shape.union(circle2)
    if surrounded:
        shapeout= Point(0,0).buffer((Rint+Rext)*1.01)
        shape=shapeout.difference(shape)
#    shape_in = shape.buffer(-Rext*0.1)
#    shape = shape.difference(shape_in)
    return shape
        
def show_shapes(shapes):
    plt.axis('equal')
    for shape in shapes:    
        shapeint = shape.buffer(-0.4)
        shape = shape.difference(shapeint)
        if type(shape) == Polygon:
            if not shape.is_empty:
                x,y = shape.exterior.xy
                plt.plot(x, y, 'k')
                plt.fill(x,y, 'gray')
                for i in shape.interiors:
                    x,y = i.xy
                    plt.plot(x, y, 'k')
                    plt.fill(x,y,'w')
                
        else :
            for sh in (shape.geoms):
                if not sh.is_empty:
                    x,y = sh.exterior.xy
                    plt.plot(x,y,'k')
                    plt.fill(x,y,'gray')
                    for i in sh.interiors:
                        x,y = i.xy
                        plt.plot(x, y, 'k')
                        plt.fill(x,y,'w')
                   

def flatten(shapes):
    new_vec = []
    for shape in shapes:
        if isinstance(shape,Polygon):
            new_vec.append(shape)
        else:
            for sh in shape.geoms:
                new_vec.append(sh)
    return new_vec

def fleur_de_lys(H,e):
    sinpi4=math.sin(math.pi/6)
    cospi4=math.cos(math.pi/6)
    sinpi6=math.sin(math.pi/12)
    cospi6=math.cos(math.pi/12)
    A = Point (-H*0.3*cospi4, H*0.0-H*0.3*sinpi4)
    B = Point ( H*0.3*cospi4, H*0.0-H*0.3*sinpi4)
    C = Point (-H*0.3*cospi6, H*0.0-H*0.3*sinpi6)
    D = Point ( H*0.3*cospi6, H*0.0-H*0.3*sinpi6)
    E = Point (H*0.5,  H*0.5)
    F = Point (-H*0.5, H*0.5)
    C1 = E.buffer(H*0.5+e*0.5)
    C2 = F.buffer(H*0.5+e*0.5)
    C3 = A.buffer(H*0.3)
    C4 = B.buffer(H*0.3)
    C5 = C.buffer(H*0.3)
    C6=  D.buffer(H*0.3)
    left = C5.difference(C3)
    right = C6.difference(C4)
    top = C1.intersection(C2)
    shape = left.union(top)
    shape = shape.union(right)
    return shape

def rosace1(N, Rint, Rmid) :
    tana = math.tan(2*math.pi/N)
    
    shape_out,shape_in = new_coffin(Rint,Rmid, alpha=360.0/2/N)
    shape2_out,shape2_in = new_coffin(Rint,Rmid, alpha=360.0/N)
    lref = tana*Rmid

    #tracing outer trifoil rosaces
    Rout = Rmid*(1+tana)
    r2 = rotate(new_rosace (Rint=lref*0.3,Rext=lref*0.2,N=4),180)
    circle = Point(0,0).buffer(Rout)
    r2 = copy.deepcopy(translate(r2,Rout,0))
    r2=r2.intersection(circle) 
 
    #tracing inner five-foil rosace
    Rext = Rmid*(1+tana*0.61)
    r3 = new_rosace(Rint=lref*0.19,Rext=lref*0.09,N=5,surrounded=True)
    r3 = copy.deepcopy(translate(r3,Rext,0))
    r3 = rotate(r3,360/2/N,(0,0))


    shape_in = rotate(shape_in,360/4/N,(0,0))
    shape_out = rotate(shape_out,360/4/N,(0,0))
    shapes2_out = rotate(shape2_out,360/2/N,(0,0))

    shapes_in = multi_rotate(shape_out,2*N)
    shapes_out = multi_rotate(shape_in,2*N)
    shapes2_out = multi_rotate(shapes2_out,N)
    shapes2 = multi_rotate(r2,N)

    r3=multi_rotate(r3,N)

    shapes_in = flatten(shapes_in)
    shapes_out = flatten(shapes_out)
    shapes2_out = flatten(shapes2_out)

    shapes = shapes_out+shapes2_out+shapes2+r3+shapes_in
 
    return shapes

def rosace2(N, Rint, Rmid, h_fleur):
    tana = math.tan(2*math.pi/N)
    shape_out,shape_in = new_coffin(Rint,Rmid, alpha=360.0/N)
    circle_in = Point (0,0).buffer(Rmid)
    
    shape_final = shape_out.difference(shape_in)
    shape_final = shape_final.difference(circle_in)


    fl =fleur_de_lys(h_fleur, h_fleur/3)
    fl = rotate(fl,90,(0,0))
    fl = translate(fl,Rmid,0)

    #tracing outer trifoil rosaces
    lref = tana*Rmid
    Rout = Rmid*(1+tana)
    r2 = rotate(new_rosace (Rint=lref*0.3,Rext=lref*0.2,N=4),180)
    circle_out = Point(0,0).buffer(Rout)
    r2 = copy.deepcopy(translate(r2,Rout,0))
    r2=r2.intersection(circle_out) 

    shape_final = rotate(shape_final,360/2/N,(0,0))

    shapes2 = multi_rotate(r2,N)
    shapes_final = multi_rotate(shape_final,N)
    shapes_fl = multi_rotate(fl,N)
    shapes_final = flatten(shapes_final)
    shapes = shapes_final+shapes2+shapes_fl
    return shapes

def ring(Rext,Rint):
    shape_out= Point(0,0).buffer(Rext)
    shape_in = Point (0,0).buffer(Rint)
    return shape_out.difference(shape_in)

def lined_ring(Rext,Rint,thickness,nwaves,ntwinned,surrounded=False):
    from math import cos,sin
    points=[]
    Rmid = 0.5*(Rext+Rint)
    delta = (Rext-Rint-2*thickness)*0.45
    for i in range (360):
        theta = i/360*2*math.pi
        R = Rmid + cos (nwaves*theta)* delta
        points.append((R*cos(theta),R*sin(theta)))
    line = LineString(points)
    line = line.buffer(thickness)
    lines = multi_rotate(line, ntwinned)

    if surrounded:
        shapeout= ring(Rext,Rext-thickness)
        lines.append(shapeout)
#
    return lines

def four_surrounding_circles(Rext,thickness,nwaves,ntwinned):
    from math import sqrt
    alpha = (3-2*sqrt(2))
    beta= (1+alpha)*sqrt(2)/2
    circle = lined_ring(alpha*Rext, alpha*Rext*0.3, thickness, nwaves, ntwinned,surrounded=True)
    circle = unary_union(circle)
    shapes=[]
    shapes.append(translate(circle,Rext*beta,Rext*beta))
    shapes.append(translate(circle,Rext*beta,-Rext*beta))
    shapes.append(translate(circle,-Rext*beta,-Rext*beta))
    shapes.append(translate(circle,-Rext*beta,Rext*beta))
    return shapes

def eight_surrounding_circles(Rext,Nfoil):
    alpha = 1.6/19*Rext
    beta = 11.2/19*Rext
    rosace = new_rosace(alpha*0.6, alpha*0.4, Nfoil,surrounded=True)
    shapes=[]
    shapes.append(translate(rosace, beta, (Rext-alpha*1.1)))
    shapes.append(translate(rosace,  (Rext-alpha*1.1), beta))
    shapes.append(translate(rosace, -beta, (Rext-alpha*1.1)))
    shapes.append(translate(rosace,  -(Rext-alpha*1.1), beta))
    shapes.append(translate(rosace, -beta, -(Rext-alpha*1.1)))
    shapes.append(translate(rosace,  -(Rext-alpha*1.1), -beta))
    shapes.append(translate(rosace, beta, -(Rext-alpha*1.1)))
    shapes.append(translate(rosace,  (Rext-alpha*1.1), -beta))
    return shapes
    
    
if __name__== "__main__" : 
    dims = [10,15,20,40,50,58,65,78]
    dims = [7,10,15,42,52,60,65,78]
    
    Router_ring = dims [5]
    circle_out = ring(Router_ring,Router_ring-1)
    shape_out = rosace1(16,dims[2],dims[3])
    shape_outer_circle = rosace2(32, dims[4],dims[6],4)
    shape_mid = lined_ring(dims[2]-1,dims[1],0.2,10,3)
    shape_mid2 = ring (dims[0]+(dims[1]-dims[0])*0.8,dims[0]+(dims[1]-dims[0])*0.2)
    shape_int = new_rosace(0.7*dims[0],0.3*dims[0],5,inner_circle_reduction_factor=0.6,surrounded=True)
    circles = four_surrounding_circles(dims[7],0.5,5,3)
    small_circles = eight_surrounding_circles(dims[7], 4)
    frame = rectangle(2*dims[7], 2*dims[7])
    show_shapes ([frame]+[circle_out]+shape_out+shape_mid+[shape_mid2]+[shape_int]+shape_outer_circle+circles+small_circles)
 
    plt.show()