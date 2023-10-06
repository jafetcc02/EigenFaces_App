
import IPython
import matplotlib
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist
import numpy as np
import cv2
from tkinter import *
from tkinter import filedialog as fd
from ipywidgets import interact, fixed, FloatSlider, IntSlider, Checkbox, FloatRangeSlider



#Vamos a estar guardando los pesos para las caras de entrenamiento y de testeo 

train_weights = {} #Entrenamiento
test_weights = {} #Para reconocimiento o clasificación
train_eigen_face_vec = {}
eigen_vec_error_dict = {}


#Función para leer las imagenes de la base 
def load_images(path):
	file_dic = {}
	sub_directory = []
	image_list = []
	id_list = []
	for subdir, dirs, files in os.walk(path): #Accedemos al folder 
		for file in files:
			split_list = file.split("_") #Lo hacemos para solo leer los primeros caracteres del nombre de la imagen 
			id = split_list[0]
			id_list.append(id)
			if file.find("jpg") > -1: #Leemos que la imagen sea de formato jpg obligatoriamente 
				if subdir in file_dic:
					file_dic[subdir].append(os.path.join(subdir, file)) #Dirección de la imagen 
					image = (cv2.imread(os.path.join(subdir, file), 0)) / 255. #Conversion rgb
					image_list.append(image) #Insetanmos en nuestra lista de imagenes 
				else:
					sub_directory.append(subdir)
					file_dic[subdir] = [os.path.join(subdir, file)]
					image = (cv2.imread(os.path.join(subdir, file), 0)) / 255.
					image_list.append(image)
	return image_list, file_dic, sub_directory, id_list




def display_images(image_list, title): #Con esta funcion mostramos o imprimimos las imagenes 

	#Ahora imprimimos las imagenes en un cuadrado de 5x5 imagenes 

	fig1, axes_array = plt.subplots(5, 5)
	fig1.set_size_inches(5, 5)
	i = 0
	for row in range(0, 5):
		for col in range(0, 5):
			image = cv2.resize(image_list[i], (100, 100))
			axes_array[row, col].imshow(image, cmap=plt.cm.gray) #Las ponemos en la escala de grises 
			axes_array[row, col].axis('off')
			i += 1
	plt.suptitle(title)
	plt.show()

#Con esta función calculamos la matriz de covarianza de la base 
def calculate_covariance(matrix):
	return np.cov(matrix, rowvar=False)

#Con esta función calculamos la matriz correspondiente a las imágenes 
def calculate_image_vector_matrix(image_list):
	# =============================================================
	#               CALCULAMOS EL VECTOR CORRESPONDIENTE A LA CARA 
	# =============================================================
	val_1, val_2 = image_list[0].shape
	rows = val_1 * val_2
	image_vec_matrix = np.zeros((rows, len(image_list)))
	i = 0
	for image in image_list:
		vector = image.flatten()
		vector = np.asmatrix(vector)
		image_vec_matrix[:, i] = vector
		i += 1
	return image_vec_matrix #retornamos la matriz 


def calculate_mean_face(image_vec_matrix):
	# =============================================================
	#			CALCULAMOS EL VECTOR DE LA CARA PROMEDIO
	# =============================================================
	mean_face_vec = np.mean(image_vec_matrix, axis=1) #Notemos que calculamos el promedio 
	mean_face_vec = np.asmatrix(mean_face_vec) #Lo convertimos a matriz 
	
	# =============================================================
	#           CALCULAMOS LA IMAGEN DE LA CARA PROMEDIO
	# =============================================================
	global mean_face_img
	mean_face_img = np.reshape(mean_face_vec, (size, size)) #Hacemos un reshape 
	
	return mean_face_vec

def store_weights(k, zero_mean_face_matrix):
	k_eigen_faces_vec = train_eigen_face_vec[k]
	# =============================================================
	#          			 CALCULAMOS LOS PESOS
	# =============================================================
	k_weights = np.dot(k_eigen_faces_vec, zero_mean_face_matrix)
	k_weights = np.transpose(k_weights)
	test_weights[k] = k_weights

def reconstruct(k, eig_vec, zero_mean_face_matrix, mean_face_vec, show = False):
	k_faces = []
	# =============================================================
	#              ELIGIMOS LOS PRIMEROS K EIGENVECTORES 
	# =============================================================
	k_eig_vec = eig_vec[0: k, :]
	
	# =============================================================
	#	CALCULAMOS LAS EIGENFACES DE LOS PRIMEROS K EINGENVECTORES
	# =============================================================
	k_eigen_faces_vec = np.dot(zero_mean_face_matrix, k_eig_vec.T)
	k_eigen_faces_vec = np.transpose(k_eigen_faces_vec)
	train_eigen_face_vec[k] = k_eigen_faces_vec
	# =============================================================
	#          				CALCULAMOS LOS PESOS 
	# =============================================================
	k_weights = np.dot(k_eigen_faces_vec, zero_mean_face_matrix)
	k_weights = np.transpose(k_weights)
	train_weights[k] = k_weights
	
	
	
	# ==============================================================
	#          		     REALIZAMOS LA RECONSTRUCCIÓN
	# ===========q==================================================
	if (show == False):
		k_reconstruction = mean_face_vec + np.dot(k_weights, k_eigen_faces_vec)
		k_reconstruction = np.transpose(k_reconstruction)
		
		for face in range(0, k_reconstruction.shape[1]):
			k_faces.append(np.reshape(k_reconstruction[:, face], (size, size)))
		
		# =============================================================
		#          		      MOSTRAMOS LAS CARAS RECONSTRUIDAS 
		# =============================================================
		display_images(k_faces, "Reconstructed Images with " + str(k) + " Eigenvectors")

def get_zero_mean_face_matrix(image_vec_matrix, mean_face_vec):
	# =============================================================
	#    Calculamos la matriz de caras sin cara promedio
	# =============================================================
	zero_mean_face_matrix = []
	count = 0
	loop_end = np.shape(image_vec_matrix)[1]
	for i in range(loop_end):
		image_col_vector = image_vec_matrix[:, i]
		image_col_vector = image_col_vector - mean_face_vec
		
		if count == 0:
			zero_mean_face_matrix = image_col_vector
			count += 1
		else:
			zero_mean_face_matrix = np.vstack((zero_mean_face_matrix, image_col_vector))
		
	zero_mean_face_matrix = zero_mean_face_matrix.T
	return zero_mean_face_matrix

def euclidean(test_weight, training_weights, train_images, test_img, k, l, acc, thr, test_id_list, train_id_list):
	# =============================================================
	#              calculamos la distancia euclideana 
	# =============================================================
	dist = []
	for weight in range(0, training_weights.shape[0]):
		dist.append(cdist(np.asmatrix(test_weight), np.asmatrix(training_weights[weight, :]), 'euclidean'))
	closest = dist.index(min(dist))
	# =============================================================
	#                   Hacemos la clasificación
	# =============================================================
	if min(dist) > thr and thr !=0 : #ES EL CASO EN EL QUE NO CLASIFICA O RECONOCE A LA PERSONA 
		if test_id_list[l] == "S099" or train_id_list[l] not in train_id_list:
			acc += 1
		if k == 25:
			#print("Eigen_Vectors Used are %d"%(k),"distance calculated is %d"% (min(dist)))
			fig1, axes_array = plt.subplots(1, 2)
			fig1.set_size_inches(5, 5)
			axes_array[0].imshow(test_img, cmap=plt.cm.gray)
			axes_array[0].axis('off')
			axes_array[0].title.set_text('Test_Image')
			axes_array[1].imshow(np.ones((425, 425)), cmap=plt.cm.gray)
			axes_array[1].axis('off')
			axes_array[1].title.set_text('Non-Face or Uknown')
            
	elif test_id_list[l] == train_id_list[closest]:
		acc += 1
		
		if k == 25:
			#print("Eigen_Vectors Used are %d"%(k),"distance calculated is %d"% (min(dist)))
			fig1, axes_array = plt.subplots(1, 2)
			fig1.set_size_inches(5, 5)
			axes_array[0].imshow(test_img, cmap=plt.cm.gray)
			axes_array[0].axis('off')
			axes_array[0].title.set_text('Test_Image')
			axes_array[1].imshow(train_images[closest], cmap=plt.cm.gray)
			axes_array[1].axis('off')
			axes_array[1].title.set_text('Classified_Image')

	elif test_id_list[l] != train_id_list[closest]:
		if  k == 25: 
			#print("Eigen_Vectors Used are %d"%(k),"distance calculated is %d"% (min(dist)))
			fig1, axes_array = plt.subplots(1, 2)
			fig1.set_size_inches(5, 5)
			axes_array[0].imshow(test_img, cmap=plt.cm.gray)
			axes_array[0].axis('off')
			axes_array[0].title.set_text('Test Image')
			axes_array[1].imshow(train_images[closest], cmap=plt.cm.gray)
			axes_array[1].axis('off')
			axes_array[1].title.set_text('Image Found')
	plt.show()
	return acc

def get_eig_vectors(zero_mean_face_matrix, mean_face_vec):
	# zero_mean_face_matrix = image_vec_matrix - mean_face_vec
	covariance = calculate_covariance(zero_mean_face_matrix)
	# =============================================================
	#      			   calculating Eigen Faces
	# =============================================================
	eig_values, eig_vectors = np.linalg.eig(covariance)
	eig_faces_vec = np.dot(zero_mean_face_matrix, eig_vectors)
	global eig_face
	eig_face = []
	for face in range(0, eig_faces_vec.shape[1]):
		eig_face.append(np.reshape(eig_faces_vec[:, face], (size, size)))

	
	# =============================================================
	#      				Sorting Eigen Values
	# =============================================================
	eig_values, eig_vectors = zip(*sorted(zip(eig_values, eig_vectors)))
	eig_values = np.asarray(list(eig_values[::-1]))
	eig_vectors = np.asarray(list(eig_vectors[::-1]))
	return eig_vectors


size = 425

#Definimos las funciones de los botones

#Subimos las imagenes de entrenamiento y realizamos el precalculo de la cara promedio y las eigencaras
def TrainImages():
	global train_path, train_img_list, train_file_dict, train_sub_dir, train_id_list 
	train_path = fd.askdirectory()
	if train_path is None:
		return
	train_img_list, train_file_dict, train_sub_dir, train_id_list = load_images(train_path)
	MeanFace()
	Eigenface()
	for i in range(len(train_img_list)):
		reconstruct(i+1, eig_vectors, train_zero_mean_face_matrix, train_mean_face_vector, show = True)
	button2.pack()
	button3.pack()
	button4.pack()
	label.pack()
	slider.pack()
	button5.pack()
	button6.pack()

def showTrainImages():
	display_images(train_img_list, "Training Images Set")

#Obtenemos la cara promedio del conjunto de entrenamiento
def MeanFace():
	global train_image_vector_matrix, train_mean_face_vector, train_zero_mean_face_matrix
	train_image_vector_matrix = calculate_image_vector_matrix(train_img_list)
	train_mean_face_vector = calculate_mean_face(train_image_vector_matrix)
	train_zero_mean_face_matrix = get_zero_mean_face_matrix(train_image_vector_matrix, train_mean_face_vector)

def showMeanFace():
	plt.imshow(cv2.resize(mean_face_img, (100, 100)), cmap='gray')
	plt.suptitle("Mean Face from Training Set")
	plt.show()

#Obtenemos las eigencaras del conjunto de entrenamiento
def Eigenface():
	global eig_vectors
	eig_vectors = get_eig_vectors(train_zero_mean_face_matrix, train_mean_face_vector)

def showEigenface():
	display_images(eig_face, "Eigenfaces")

#Reconstruimos las caras con los eigenvectores
def ImageRecon():
	reconstruct(slider.get(), eig_vectors, train_zero_mean_face_matrix, train_mean_face_vector)

#Funciones para realizar Face ID
def load_image(name):
    id_list = []
    image_list = []
    split_list = name.split("_")
    id = split_list[0]
    id_list.append(id)

    image = (cv2.imread(name, 0)) / 255
    image_list.append(image)

    return image_list, id_list


def classifying1(thr, image_name):

	
	test_img_list, test_id_list = load_image( image_name)

	test_image_vector_matrix = calculate_image_vector_matrix(test_img_list)
	test_zero_mean_face_matrix = get_zero_mean_face_matrix(test_image_vector_matrix, train_mean_face_vector)
	for i in range(0, len(train_img_list)):
		store_weights(i+1, test_zero_mean_face_matrix)

	acc = 0
	acc = euclidean(test_weights[12+1][0, :], train_weights[12+1], train_img_list, test_img_list[0], 25, 0, acc, thr,  test_id_list, train_id_list) 

#Obtenemos una imagen y le aplicamos FaceID para encontrar la mas parecida en el conjunto de entrenamiento
def ID():	
	filename = fd.askopenfilename()
	if filename is None:
		return
	classifying1(5000, filename)


#Definimos los sliders y botones de la gui
root = Tk()
root.title("EigenFaces")
root.geometry("400x300")

button1 = Button(root, text="Load New Training Images", command = TrainImages)
button2 = Button(root, text="Show Training Images", command = showTrainImages)
button3 = Button(root, text="Show Mean Face", command = showMeanFace)
button4 = Button(root, text="Show Eigenfaces", command = showEigenface)
button5 = Button(root, text="Show Image Reconstruction", command = ImageRecon)
slider = Scale(root, from_= 2, to=25, orient=HORIZONTAL)
slider.set(2)
label = Label(root,text="# of Eigenvectors for Reconstruction")
button6 = Button(root, text="Upload Image for Face ID", command = ID)

button1.pack()


root.mainloop()