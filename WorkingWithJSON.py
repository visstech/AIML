
import json
import pdb # python debugger 


class Student:
    intitute = 'Gvi'
    def __init__(self,name,rollno,course,marks): 
        self.name =  name
        self.rollno = rollno
        self.course = course
        self.marks = marks 
    def calculate_percentage(self,maxmarks):
        return round(self.marks*100/maxmarks,2)
    
S1 = Student('senthil',101,'data science',140)
S2 = Student('Thalir',102,'data science',120)
pdb.set_trace()  # stop the execution here. for tracing n for execute next line of code c for continue untill end.
S3 = Student('Aruthra',103,'data science',130)
S4 = Student('Kurthikhashini',104,'data science',110)
S5 = Student('varun',105,'data science',130)
S6 = Student('vimal',106,'data science',140)

Students = [S1.__dict__,S2.__dict__,S3.__dict__,S4.__dict__,S5.__dict__,S6.__dict__] # this will create dictionary of all the objects
 

print(Students)

with open('Myjson.json','w') as f: #writing the data as json file.
    json.dump(Students,f)
    
with open('Myjson.json','r') as f: # reading from json file.
    data = json.load(f)

print('Reading from the Myjson file:\n')
print(data)

'''  
class Student:
    intitute = 'Gvi'
    def __init__(self) -> None:
        self.name = input('Enter the name ')
        self.roll = input('Enter the roll no:')
    def courses(self):
        self.course = input('Enter the couse name')    
s1 = Student()
s2 = Student()
s2.courses()

print(f" Institue = {s1.intitute} name = {s1.name}  roll no = {s1.roll} ")
print(f" Institue = {s2.intitute} name = {s2.name}  roll no = {s2.roll} ,Course = {s2.course}")
'''


class Student:
    intitute = 'Gvi'
    def __init__(self,name,rollno,course,marks): 
        self.name =  name
        self.rollno = rollno
        self.course = course
        self.marks = marks 
    def calculate_percentage(self,maxmarks):
        return round(self.marks*100/maxmarks,2)
    
s1 = Student('senthilkumar',101,'DA',120)      
print(f"Name = {s1.name}, Roll No: {s1.rollno} ,course :{s1.course} marks:{s1.marks} Percentage : {s1.calculate_percentage(140)}") 


class car():
    def __init__(self,make,model,year):
        self.make = make
        self.model = model
        self.year  = year
    def start(self):
        return "The car is starting"
    def stop(self):
        return "The car is stopping"
    def get_info(self):
        return  f" Make is {self.make}, Mode is {self.model} year is:{self.year}"  
    
cars = car("Toyota", "Camry", 2020)   
print(f"{cars.start(),cars.stop(),cars.get_info()}") 

