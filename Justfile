# Variable from CMakeLists
PROJECT_NAME := `grep -oP 'set\(PROJECT_NAME \K[^\)]+' CMakeLists.txt`

# list available commands
default: 
	just -l

# initialize the project with cmake
init:
	mkdir -p ./build && cmake --preset default -S `pwd`/ -B `pwd`/build

# rebuild with cmake
build:
	cmake --build ./build

# run generated binary
run:
	./build/{{PROJECT_NAME}}

test:
	./build/tests
	
# rm generated binary
clean:
	rm -rf ./build 

# re-{clean,build,run}
re: clean build run test

# visualize the dependency graph
deps:
	python ./tools/deps.py

# check setup
check:
	./tools/check_setup.sh