# Variable from CMakeLists
PROJECT_NAME := `grep -oP 'set\(PROJECT_NAME \K[^\)]+' CMakeLists.txt`

# list available commands
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
	./build/{{PROJECT_NAME}}

test:
	./build/tests
	
# rm generated binary
clean:
	rm ./build/{{PROJECT_NAME}}

# re-{clean,build,run}
re: clean build run test

# visualize the dependency graph
deps:
	python ./tools/deps.py

# check setup
check:
	./tools/check_setup.sh