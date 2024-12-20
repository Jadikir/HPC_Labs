В данной лабораторной используется два подхода для выполнения умножения матриц(в .cu файле GPU, в cpp - OpenMp): один с использованием CUDA для GPU, а другой с использованием OpenMP.

На CUDA:

реализовано ядро matrixMulKernel, которое выполняет умножение матриц. Основные шаги распараллеливания включают:

![image](https://github.com/user-attachments/assets/266a280c-cd61-4b52-a17b-3dd6205444af)

Каждый поток CUDA отвечает за вычисление одного элемента выходной матрицы. Это достигается путем определения уникальных индексов row и col на основе идентификаторов блока и потока.

Все потоки выполняют вычисления одновременно, что позволяет существенно уменьшить время выполнения по сравнению с последовательным подходом. Каждый поток вычисляет свою часть результирующей матрицы, что эффективно использует архитектуру GPU.

Выделяем память на девайсе(Видеокарта)->Копируем данные с процессора на GPU(на процессоре матрицы заполнялись)->выполняем перемножение-> возвращаем данные обратно на процессор и сравниваем с последовательным перемножением.

OpenMp:

![image](https://github.com/user-attachments/assets/044173a4-fd4d-4dd7-8bfe-662aa2ac93db)

Простейшее распаралеливание с использованием OPENMP сделано в целом для более точного сравнения

Результаты:

![image](https://github.com/user-attachments/assets/abdae47d-28d1-4a0c-a5ae-588b6d5471c9)

При очень маленьких значениях -100x100 последовательный алгоритм лучше, потому что прирост от распараллеливания не превышает затрат на осуществление этого распараллеливания.

500x500 тут уже лучшим является параллельный на CPU так,как уже затраты на распараллеливание не так значительны как прирост производительности. Но вот с GPU немного иначе, там затрат больше, так как приходится туда сюда копировать и "реальная скорость" этого перемножения намного быстрее чем процессорные операции.

1000X1000 здесь та же ситуация, но уже больше заметна тенденция - чем больше операций, тем больше выигрыш у GPU 

2000X2000 уже значительно заметно ускорение от GPU.
