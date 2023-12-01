import Program_Functions
import time

def initial():
    time.sleep(2)
    print("\n####\nFor Registering face, Enter 1  \nFor Attendance, Enter 2 \nFor Exit, Enter 0\n####\n")
    a1=input('Give your input : ')
    print()
    if a1=='1':
        if Program_Functions.Input():
            return initial()
    elif a1=='2':
        if Program_Functions.face_rec():
            return initial()
    elif a1=='0':
        print("Thank You! See you Soon...\n")
        exit
    else:
        print("#### Please Give The Correct Value ####")
        return initial()

print("\n  \"Introduction to Engineering Project\"")
print("---Facial Recognition Attendance System---")
initial()