import tkinter
import tkinter.filedialog
import tkinter.ttk
from PIL import Image,ImageTk
from torchvision import transforms as transforms
# from test import main, model
from test import main

# 创建UI
win = tkinter.Tk()
win.title("picture process")
win.geometry("1280x1080")
win.configure(bg="#F0F8FF")

# 创建和放置标签
tkinter.Label(win, text="Original Picture", bg="yellow", fg='black', font=("Helvetica", 16)).place(x=200, y=50)
tkinter.Label(win, text="Transformation!", bg="yellow", fg='black', font=("Helvetica", 16)).place(x=900, y=50)

# 声明全局变量
original = Image.new('RGB', (300, 400))
save_img = Image.new('RGB', (300, 400))
count = 0
e2 = None
e2 = str(e2)
file_name = None
img2 = tkinter.Label(win)


def choose_file():
	'''选择一张照片'''
	select_file = tkinter.filedialog.askopenfilename(title='select the picture')
	global file_name
	file_name=select_file
	e.set(select_file)
	load = Image.open(select_file)
	load = transforms.Resize((400,400))(load)
	# 声明全局变量
	global original
	original = load
	render = ImageTk.PhotoImage(load)
	img  = tkinter.Label(win,image=render)
	img.image = render
	img.place(x=100,y=100)

def coloring():
    '''图片生成'''
    # model()
    new_img = Image.open('generate.png')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)

    global img2
    img2.destroy()
    img2  = tkinter.Label(win,image=render)
    img2.image = render
    img2.place(x=800,y=100)

def transfer():
    main(file_name)
    # model()
    new_img = Image.open('generate.png')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)

    global img2
    img2.destroy()
    img2  = tkinter.Label(win,image=render)
    img2.image = render
    img2.place(x=800,y=100)

def edge_detect():
    '''边缘检测'''
    main(file_name)
    new_img = Image.open('canny&HED.jpg')
    new_img = transforms.Resize((400,400))(new_img)
    render = ImageTk.PhotoImage(new_img)

    global img2
    img2.destroy()
    img2  = tkinter.Label(win,image=render)
    img2.image = render
    img2.place(x=800,y=100)



e = tkinter.StringVar()
e_entry = tkinter.Entry(win, width=68, textvariable=e, font=("Helvetica", 12))
e_entry.place(x=200, y=650)
# e_entry.pack()

custom_font = ("Arial", 12)
button_style = "TButton"
# 按钮样式
button_style = tkinter.ttk.Style()
button_style.configure("Custom.TButton", font=custom_font, background="red", foreground="black")

# 按钮
tkinter.ttk.Button(win, text="Select", command=choose_file, style="Custom.TButton").place(x=50, y=700)
tkinter.ttk.Button(win, text="Edge detect", command=edge_detect, style="Custom.TButton").place(x=400, y=700)
tkinter.ttk.Button(win, text="Coloring", command=coloring, style="Custom.TButton").place(x=700, y=700)
tkinter.ttk.Button(win, text="Style transfer", command=transfer, style="Custom.TButton").place(x=1000, y=700)
tkinter.ttk.Button(win, text="Exit", command=win.quit, style="Custom.TButton").place(x=550, y=750)

# # 文件选择
# button1 = tkinter.Button(win, text ="Select", command=choose_file, width=20, height=2, font=custom_font, style = "TButton").place(x=50, y=700)
# # button1.pack()
#
# button2 = tkinter.Button(win, text="edge detect" , width=20, height=2, font=custom_font, style = "TButton").place(x=400, y=700)
# # button2.place(x=570,y=200)
#
# button3 = tkinter.Button(win, text="coloring" , width=20, height=2, font=custom_font,style = "TButton").place(x=700, y=700)
# # button3.place(x=570,y=300)
#
# button4 = tkinter.Button(win, text="style transfer" , width=20, height=2, font=custom_font,style = "TButton").place(x=1000, y=700)
# # button4.place(x=570,y=400)

img = tkinter.Label(win, text="无图片", bg="#f0f0f0")
img.place(x=100, y=100)
img2 = tkinter.Label(win, text="无图片", bg="#f0f0f0")
img2.place(x=800, y=100)

# 退出按钮
# button0 = tkinter.Button(win,text="Exit",command=win.quit, width=20, height=2, font=custom_font,button_style = "TButton").place(x=600, y=800)
# button0.place(x=570,y=650)
win.mainloop()
