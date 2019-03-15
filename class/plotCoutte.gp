reset
#set autoscale
set key left Right nobox samplen 7 lmargin at -2.2,0.95 font ",20"
#set key left nobox samplen 10
#set key outside horizontal bottom center
set title "Coutte Flow" font ",20"
set xlabel "y/H" font ",20"
set ylabel "u/U_p" font ",20"
set tics font ",20"
plot "coutte.dat" using 2:1 title 'Exact' with linespoints lw 2 pt 28 ps 2, \
	"coutte.dat" using 3:1 title 'Numerical' with lines lw 2
#set terminal svg size 1600,1200 font ",40"
#set output "sample.svg"
#set terminal jpg color enhanced "Helvetica" 20
set output "output.jpg"
replot
set term x11

