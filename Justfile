default: 
	just -l

# init the project ./build cmake
init:
	mkdir -p ./build && cd ./build && cmake ..

# rebuild with cmake
build:
	cmake --build ./build

# run generated binary
run:
	./build/cpp_template

test:
	./build/tests
	
# rm generated binary
clean:
	rm ./build/cpp_template

# re-{clean,build,run}
re: clean build run test
