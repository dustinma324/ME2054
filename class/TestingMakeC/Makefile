PRG = heatSolve
EXE = $(PRG).exe

CC = gcc
DEBUG = -g
CFLAGS = -Wall -O2 -std=c99 $(DEBUG)
LDFLAGS = 
LIBS = -lm

SRCS = 	quiz_2_heat.c	exactSolution.c\
	meshGrid.c	solveHeat_1D.c\
	writeOutput.c

DEPS =	exactSolution.h	meshGrid.h\
	solveHeat_1D.h	writeOutput.h\
	parameter.h

OBJECTS = $(SRCS:.c=.o)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXE): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean: 
	rm -f *.o $(EXE) core

.PHONY: tar

tar:
	tar -cvf $(PRG).tar $(SRCS) $(DEPS) Makefile
	gzip $(PRG).tar
