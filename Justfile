# Variable from CMakeLists
PROJECT_NAME := `grep -oP 'set\(PROJECT_NAME \K[^\)]+' CMakeLists.txt`

# list available commands
default: 
	just -l

# initialize the project with cmake
init:
    git submodule update --init --remote --recursive

# rebuild with cmake
build *args="default":
    cmake --preset={{args}}
    cmake --build --preset={{args}}

# run generated binary
run:
	./build/{{PROJECT_NAME}}

test:
	./build/tests
	
# reset the build environment to clean state
reset:
	rm -rf ./build && mkdir -p ./build

# re-{build,run,test}
re: build run test

# visualize the dependency graph
deps:
	python ./tools/deps.py

# check setup
check:
	./tools/check_setup.sh

# concat project into 1 file
concat *args:
    ./tools/concat.sh {{args}}
