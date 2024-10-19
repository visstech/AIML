
'''
No_of_employee = int(input('Enter the number of employee:'))
count = 0; 
empolyee_data = ''
emp_list =[]
platinum_employee =[]
print(No_of_employee)
while No_of_employee > 0:
    empolyee_data = input('Enter the employe name,age,email, salary :')
    emp = empolyee_data.split(' ') 
    emp_list.extend(emp)
    No_of_employee = No_of_employee - 1;

for employee in range(len(emp_list)) :
    print(emp[3])
    platinum_employee.append(emp[3])
    with open('my_text.txt','w') as f:
        f.write(emp[3]) 
    
full_name = lambda f_name,l_name : f_name + " " + l_name

print(full_name('senthil','kumar'))

''' 
marks = [35,67,12,67,7,22,56,74,3,2,]
print(list(filter(lambda a : a>= 25,marks)))

output = list(map(lambda a : a+5,marks))

names = ['senthil','kavitha','thalir','ram senthil']
name = 'sentil'
print(name in 'sen')
print(list(filter(lambda a : a in 'sen',names)))

 
marks = input('Enter marks of 10 students :').split(' ')
 
marks = list(filter(lambda a: a> 40,list(map(lambda a : round(float(a)*100 / 55,2),marks))))
print(marks)
import pandas as pd 
print(pd.__version__)
