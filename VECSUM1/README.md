CPU:

Функция sum_cpu использует стандартную библиотечную функцию std::accumulate для последовательного суммирования элементов вектора. Это выполняется на одном ядре процессора.

![image](https://github.com/user-attachments/assets/8d2cbb91-6574-47f3-a0d4-bb526e7e642c)

GPU:

Вся задача суммирования разделяется на независимые подзадачи, каждая из которых выполняется в своем блоке на GPU. В блоке каждый поток получает свой элемент данных для суммирования.

В каждом блоке используется стратегия редукции (суммирование данных с разделением на шаги), при которой на каждом шаге потоки суммируют данные с соседними потоками до тех пор, пока не останется одна сумма на блок. Эта сумма записывается в глобальную память в выходной массив.

![image](https://github.com/user-attachments/assets/de2f4bf8-c703-47db-a594-7b8e5262efcc)

Распараллеливание:

Весь массив делится на блоки по 256 элементов, каждый блок обрабатывается независимо.

Каждый поток в блоке загружает один элемент данных в shared memory и участвует в редукции (суммировании) с логарифмическим количеством шагов. Это позволяет суммировать элементы в блоке за минимальное количество итераций.

После завершения редукции внутри каждого блока промежуточные суммы блоков сохраняются в массив output в глобальной памяти. Если количество блоков превышает единицу, то запускается еще одна фаза редукции над этим массивом сумм блоков.

![image](https://github.com/user-attachments/assets/9784feaf-d4f1-487a-b728-07ce6a1e82fa)

Ускорение в данном случае CPU/GPU:

почему такие результаты?

Малые данные (1000–100 000) :  Накладные расходы на инициализацию GPU и копирование данных выше, чем выигрыш от параллелизма.

Средние данные (100 000 – 1 000 000) : GPU начинает показывать свое преимущество, однако выигрыш не столь велик из-за накладных расходов.

Очень большие данные (1 000 000 000): Ускорение от распараллеливания начали давать плоды и появился значительный выигрыш.
