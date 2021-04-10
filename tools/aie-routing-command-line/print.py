import math

from enum import Enum

class Direction(Enum):
    Horz = 0;
    Vert = 1;


class canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.vert_line_list = []
        self.horz_line_list = []
        self.characters = []
        
    def direction(self, line) :
        Horz_Stationary = False;
        Vert_Stationary = False;
        
        if (line[0][0] == line[1][0]):
            # Line Stationary in Horz Axis
            Horz_Stationary = True
        if (line[0][1] == line[1][1]):
            # Line Stationary in Vert Axis
            Vert_Stationary = True

        if (Horz_Stationary and Vert_Stationary):
            # Crash
            raise Exception("Line is Diagonal");
        if (not Horz_Stationary and not Vert_Stationary):
            # Crash
            raise Exception("Line is a dot");
        if ((not Horz_Stationary) and Vert_Stationary):
            return Direction.Horz;
        
        if (Horz_Stationary and (not Vert_Stationary)):
            return Direction.Vert;
        
    def draw_character(self, point, character):
        self.characters.append([point, character]);
        
    def draw_line(self, start, finish):

        if (self.direction([start,finish]) == Direction.Vert):
            self.vert_line_list.append([start, finish])
        else:
            self.horz_line_list.append([start, finish])
        
    def draw_square(self, center, size):
        horz_origin = math.floor((center[0] + 0.5) - (size/2));
        horz_extent = math.ceil((center[0] + 0.5) + (size/2));

        vert_origin = math.floor( (center[1] + 0.5)- (size/2));
        vert_extent =  math.ceil( (center[1] + 0.5) + (size/2));
        
        top_left = (horz_origin, vert_origin );
        top_right = (horz_extent, vert_origin);
        bottom_left = (horz_origin, vert_extent);
        bottom_right = (horz_extent, vert_extent);
        
        self.draw_line(top_left, top_right)
        self.draw_line(top_right, bottom_right)
        self.draw_line(bottom_left, bottom_right)
        self.draw_line(top_left, bottom_left)

    def vertical_index(self,point):
        return point[1];
    def horizontal_index(self,point):
        return point[0];
        
    def within_line(self, point, line):
        Horz_Stationary = False;
        Vert_Stationary = False;
        
        if (line[0][0] == line[1][0]):
            # Line Stationary in Horz Axis
            Horz_Stationary = True
        if (line[0][1] == line[1][1]):
            # Line Stationary in Vert Axis
            Vert_Stationary = True

       # print("HS: {}, VS: {}, {}".format(Horz_Stationary, Vert_Stationary, line));
            
        if (Horz_Stationary and Vert_Stationary):
            # Crash
            raise Exception("Line is Diagonal");
        if (not Horz_Stationary and not Vert_Stationary):
            # Crash
            raise Exception("Line is a dot");
        
        if (Horz_Stationary and (not Vert_Stationary)):
            # it's a vertical line
            # Sort the tuples by Horz
            line.sort(key=self.vertical_index);
            start_line = ( (point[1] == line[0][1]) and (point[1] <= line[1][1]) and (point[0] == line[0][0]) )
            in_line = ( (point[1] > line[0][1]) and (point[1] < line[1][1]) and (point[0] == line[0][0]) )
            end_line = ( (point[1] > line[0][1]) and (point[1] == line[1][1]) and (point[0] == line[0][0]) )
            
            #print("Vert {}, point {}, {}".format(line, point, in_line));
            
        if ((not Horz_Stationary) and Vert_Stationary):
            # it's a horizonal line
            line.sort(key=self.horizontal_index)
            start_line = ( (point[0] == line[0][0]) and (point[0] <= line[1][0]) and ( point[1] == line[0][1]))
            in_line = ( (point[0] > line[0][0]) and (point[0] < line[1][0]) and ( point[1] == line[0][1]))
            end_line = ( (point[0] > line[0][0]) and (point[0] == line[1][0]) and ( point[1] == line[0][1]))
            #print("Horz {}, point {}, {}".format(line, point, in_line));

        return (start_line, in_line, end_line)
    
    def find_horz_index(self,line_points):
        index =0;

        if (line_points[1]):
            index+=2
        else:
            if (line_points[0]):
                index += 1;
            if (line_points[2]):
                index +=4
        return index;

    def transform(self, index):

        # if horz through, clear bits for horz end
        # if vert through, clear bits for vert end
        
        
        
        chars = {
            0 : ' ',
            2 : u'\u2500',
            16 : u'\u2502',
            9 : u'\u250c',
            33 : u'\u2514',
            12 : u'\u2510',
            36 : u'\u2518',
            1 : u'\u2576',
            8 : u'\u2577',
            32 : u'\u2575',
            4 : u'\u2574',
            18 : u'\u253c',
            25 : u'\u251c',
            17 : u'\u251c',
            20 : u'\u2524',
            10 : u'\u252c',
            34 : u'\u2534',
            21 : u'\u253c',
            42 : u'\u253c'
            
            
        }
        try:
         char = chars[index];
        except KeyError:
         char = "x"
         
        return char;

    def combine(self, a, b):
        return [ a[0] or b[0], a[1] or b[1], a[2] or b[2]];
    
    def rasterize(self):
       for x in range(self.height):
           for y in range(self.width):
               char = "({},{})".format(y,x);
               index = 0;
               horz_line_points = [False, False, False];
               vert_line_points = [False, False, False];

               for charloc in self.characters:
                   if (charloc[0][0] == y and charloc[0][1] == x):
                       print(charloc[1], end='', sep='');
                       index = -1;
                       break
               if index == 0: 
                   for line in self.horz_line_list:
                       horz_line_points = self.combine(horz_line_points, self.within_line((y,x),line));

                   index += self.find_horz_index(horz_line_points);

                   for line in self.vert_line_list:
                       vert_line_points = self.combine(vert_line_points, self.within_line((y,x),line));

                   index += 8*self.find_horz_index(vert_line_points);

                   print("{}".format(self.transform(index)), end='', sep='')
                   #print(" {} ".format(index), end='', sep='')
   
           print("")

def draw_switchbox(canvas, xoffset, yoffset):
    c.draw_square((xoffset+5,yoffset+4),2);
   # c.draw_line((1,1), (1,5))
   
    c.draw_character((xoffset+2,yoffset+4), 5)
    c.draw_character((xoffset+9,yoffset+4), 6)

    c.draw_character((xoffset+2,yoffset+5), 5)
    c.draw_character((xoffset+9,yoffset+5), 6)

    #c.draw_character((1,4), 5)
    c.draw_character((xoffset+5,yoffset+2), 0)
    c.draw_character((xoffset+6,yoffset+2), 1)
    
    c.draw_character((xoffset+5,yoffset+7), 0)
    c.draw_character((xoffset+6,yoffset+7), 1)
    
    c.draw_line((xoffset+1,yoffset+5), (xoffset+4,yoffset+5))
    c.draw_line((xoffset+1,yoffset+4), (xoffset+4,yoffset+4))
    
    c.draw_line((xoffset+7,yoffset+5), (xoffset+10,yoffset+5))
    c.draw_line((xoffset+7,yoffset+4), (xoffset+10,yoffset+4))
    
    c.draw_line((xoffset+5,yoffset+1), (xoffset+5,yoffset+3))
    c.draw_line((xoffset+6,yoffset+1), (xoffset+6,yoffset+3))

    c.draw_line((xoffset+5,yoffset+6), (xoffset+5,yoffset+8))
    c.draw_line((xoffset+6,yoffset+6), (xoffset+6,yoffset+8))

def draw_herd(c,xoff, yoff):
    draw_switchbox(c,xoff+6,yoff+15)
    draw_switchbox(c,xoff+6,yoff+10)
    draw_switchbox(c,xoff+13,yoff+15)
    draw_switchbox(c,xoff+13,yoff+10)
    
    draw_switchbox(c,xoff+20,yoff+15)
    draw_switchbox(c,xoff+20,yoff+10)
    draw_switchbox(c,xoff+27,yoff+15)
    draw_switchbox(c,xoff+27,yoff+10)

    draw_switchbox(c,xoff+6,yoff+5)
    draw_switchbox(c,xoff+6,yoff+0)
    draw_switchbox(c,xoff+13,yoff+5)
    draw_switchbox(c,xoff+13,yoff+0)
    
    draw_switchbox(c,xoff+20,yoff+5)
    draw_switchbox(c,xoff+20,yoff+0)
    draw_switchbox(c,xoff+27,yoff+5)
    draw_switchbox(c,xoff+27,yoff+0)
    
if __name__ == '__main__':
    c = canvas(80,30);

    draw_herd(c,0,0)
    draw_herd(c,28,0)

    c.rasterize();

    
  #  print(u'\u2500') # Horz
  #  print(u'\u2502') # Vert
  #  print ("");
    # print(u'\u2574') # West
    # print(u'\u2575') # South
    # print(u'\u2576') # East
    # print(u'\u2577') # North
    # print ("");
    
    # print(u'\u2510') # West-South
    # print(u'\u2514') # East-North
    # print(u'\u250c') # South-East
    # print(u'\u2518') # West-North

    # print(u'\u252c') # tee-South
    # print(u'\u2524') # tee-west
    # print(u'\u251c') # tee-east
    # print(u'\u2534') # tee-north
    
    # print(u'\u253c') # cross

# draw a box, centered on

# draw_box(width, height)
# draw_line();
# draw_line();


# Canvas origin is top left. 

# rasterize(canvas, object_list)


#horz-end-left
#horz-end-right
#horz-through

#vert-end-left
#vert-end-right
#vert-through



#for x in canvas.width
# for y in canvas.height
  # Interrogate each object to see if it has a 
