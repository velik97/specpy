class FrameData:
	"""
	FrameData это класс, который содержит всю информацию о наборе фреймов
	"""

	WAVE_AXIS = 2

	def __init__(self, values, wave_coefs = None, description = ''):
		""" 
			Данный метод вызывается при создании экземпляра класса
			Args:
				values (numpy.ndarray): Список значений интенсивности, приведенный в виде многомерного массива numpy
				wave_coefs (double array): Список полиномиальных коэффициентов от 0 до 5 включительно для длин волн. то есть длина волны под номером i равна p0 + p1*i + p2*i*i + ...
				description (str): Описание фрейма, по умолчанию ''
		"""
		self.values = values
		self.wave_coefs = wave_coefs
		self.wave_values = FrameData.wave_lengths_from_coefs(wave_coefs, values.shape[self.WAVE_AXIS])
		self.description = description

	def __str__(self):
		"""
			Данный метод возвращает строковое описание объекта (при вызове print(), str() и тд)
		"""
		result = ""
		if self.description != "":
			result += self.description + "\n\n"

		result += "Размерность данных:" +\
			"\n  x_dim (Колличество кадров):                  " + str(self.values.shape[0]) +\
			"\n  y_dim (Колличетсво пространственных точек):  " + str(self.values.shape[1]) +\
			"\n  w_dim (Колличетсво длин волн):               " + str(self.values.shape[2])

		result += "\n\nДиапозон длин волн: " + str(self.wave_values[0]) + " - " + str(self.wave_values[-1])

		return result

	def wave_lengths_from_coefs(wave_coefs, wave_count):
		"""
		Данный метод считает массив длин волн по заданым полиномиальным коэффициентам
		Args:
			wave_coefs (numpy.ndarray): полиномиальные коэффициенты
			wave_count (int): количество длин волн

		Returns:
			(numpy.ndarray): список расчитанных длин волн
		"""
		import numpy as np
		from numpy.polynomial.polynomial import polyval

		return polyval(np.arange(wave_count), wave_coefs)

	def displaced_wave_coefs(wave_coefs, displacement):
		"""
		Данный метод пересчитывает полиномиальные коэффициенты при данном смещении начального номера длины волны, используя пирамиду Пифагора
		Args:
			wave_coefs (numpy.ndarray): Изначальные полиномиальные коэффициенты
			displacement (float): Смещение начального номера длины волны

		Returns:
			(numpy.ndarray): Новые коэффициенты
		"""
		pythagorean_pyramid=[[1],[1,1]]
		for i in range(2,len(wave_coefs)):
			l=[1]
			for y in range(i-1):
				l.append(pythagorean_pyramid[i-1][y]+pythagorean_pyramid[i-1][y+1])
			l.append(1)
			pythagorean_pyramid.append(l)

		new_coefs = []

		for i in range(len(wave_coefs)):
			new_coefs.append(0)
			for j in range(i, len(wave_coefs)):
				new_coefs[i] += wave_coefs[j]*pythagorean_pyramid[j][i]*(displacement**(j-i))

		return new_coefs

	def contains_wave(self, wave):
		"""
			Попадает ли данная волна в диапозон длин волн этого FrameData
			Args:
				wave (float): Длина волны, о которой мы хотим узнать, попадает ли она в диапозон длин волн этого FrameData

			Returns:
				(bool): True - попадает, False - нет
		"""
		return self.wave_values[0] <= wave and wave <= self.wave_values[-1]

	def wave_slice(self, min_wave_length, max_wave_length, fold = False):
		"""
			Выделяет из всех значений только те, что вписываются в заданный диапозон длин волн. Возвращает новый FrameData, исходный не трогается

		Args:
			min_wave_length (float): Минимальная длина волны для диапозона
			max_wave_length (float): Максимальная длина волны для диапозона
			fold (bool): Если True, то после выделения диапазона все значения вдоль оси длин волн сложаться (заменятся на среднее), по умлочанию False

		Returns:
			(FrameData): Объект полученый после выделения диапазона
		"""
		import numpy as np

		min_wave_num = 0
		max_wave_num = 0

		while self.wave_values[min_wave_num] < min_wave_length:
			min_wave_num += 1
			max_wave_num += 1

		min_wave_num -= 1

		while self.wave_values[max_wave_num] < max_wave_length:
			max_wave_num += 1

		max_wave_num += 1
		
		new_values = self.values

		new_values = np.swapaxes(new_values, 0, self.WAVE_AXIS)
		new_values = new_values[min_wave_num:max_wave_num]
		new_values = np.swapaxes(new_values, 0, self.WAVE_AXIS)

		new_coefs = FrameData.displaced_wave_coefs(self.wave_coefs, min_wave_num)
		new_fd = FrameData(new_values, new_coefs)

		if fold:
			new_fd = new_fd.wave_fold()

		return new_fd

	def wave_fold(self):
		"""
			Складывает все значения вдоль оси длин волн (заменяет на среднее). Возвращает новый FrameData, исходный не трогается
		"""
		import numpy as np

		new_values = np.mean(self.values, self.WAVE_AXIS)
		new_values = np.expand_dims(new_values, self.WAVE_AXIS)
		result_fd = FrameData(new_values, [self.wave_coefs[0], 0.0, 0.0, 0.0, 0.0, 0.0])

		result_fd.wave_values = [self.wave_values[0], self.wave_values[-1]]

		return result_fd

def load(path):
	"""
	Зыгружает файл типа .spe по пути
	Args:
		path (string): Путь к файлу из текущей дериктории

	Returns:
		(FrameData): Объект, содержащий все данные о списке фреймов
	"""
	with open(path, 'rb') as binary_file:
		import numpy as np
		import struct

		binary_file.seek(42) # По документации .spe здесть расположен xDim (пространственная ось)
		w_dim = struct.unpack('H', binary_file.read(2))[0]

		binary_file.seek(656) # По документации .spe здесть расположен yDim (ось длин волн)
		y_dim = struct.unpack('H', binary_file.read(2))[0]

		binary_file.seek(1446) # По документации .spe здесть расположен NumFrames (ось кадров)
		x_dim = struct.unpack('i', binary_file.read(4))[0]

		wave_coefs = []

		binary_file.seek(3263)
		for c in range(6):
			wave_coefs.append(struct.unpack('d', binary_file.read(8))[0])

		values = np.empty((x_dim, y_dim, w_dim))

		binary_file.seek(4100)

		for x in range(x_dim):
			for y in range(y_dim):
				for w in range(w_dim):
					values[x,y,w] = struct.unpack('f', binary_file.read(4))[0]

		return FrameData(values, wave_coefs, path.split(".")[-2].split("/")[-1])


def save(fd, path):
	"""
	Сохраняет файл типа .spe по пути
	Agrs:
		path (string): Путь у файлу из текущей дериктории
		fd (FrameData): Объект, данные которого будут сохранены
	"""
	from os.path import isfile
	if isfile(path):
		print("Файл с таким именем уже существует, используйте другое имя")
		return

	with open(path, 'wb+') as binary_file:
		import struct

		array = bytearray(4100 + fd.values.shape[0] * fd.values.shape[1] * fd.values.shape[2] * 4)

		x_dim = fd.values.shape[0]
		y_dim = fd.values.shape[1]
		w_dim = fd.values.shape[2]

		struct.pack_into('H', array, 42, w_dim)
		struct.pack_into('H', array, 656, y_dim)
		struct.pack_into('i', array, 1446, x_dim)

		for c in range(6):
			place = 3263 + c*8
			struct.pack_into('d', array, place, fd.wave_coefs[c])

		for x in range(x_dim):
			for y in range(y_dim):
				for w in range(w_dim):
					place = 4100 + (x*y_dim*w_dim + y*w_dim + w)*4
					struct.pack_into('f', array, place, fd.values[x,y,w])

		binary_file.write(array)

	return		