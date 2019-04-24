
#reduce: main.cu *.h
reduce: main.cu 
	nvcc -O3 main.cu -o reduce -arch=sm_70

clean:
	rm -f reduce

