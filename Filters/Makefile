CC = gcc

CFLAGS = -Wall -Wextra -O2 -std=c11

TARGET = filter

OBJS = filter.o Butterworth.o 


$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lfftw3 -lm


filter.o: filter.c Butterworth.h 
	$(CC) $(CFLAGS) -c filter.c


Butterworth.o: Butterworth.c Butterworth.h
	$(CC) $(CFLAGS) -c Butterworth.c


clean:
	rm -f $(OBJS) $(TARGET) 
