SRCS := main.cu
INCLUDE := -I.
# LDFLAGS := -L$(PARENT_DIR)particle_system/glfw/build/src `pkg-config --libs glfw3` -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXi -ldl
NAME := vector_addition
CXX := nvcc

all: $(NAME)

$(NAME): $(SRCS)
	$(CXX) $(SRCS) $(INCLUDE) -o $(NAME)

clean:
	rm -rf $(NAME)

re: clean all

.PHONY: all clean re
