SRCS := main.cu
INCLUDE := -I.
NAME := vector_addition
CXX := nvcc

all: $(NAME)

$(NAME): $(SRCS)
	$(CXX) $(SRCS) $(INCLUDE) -o $(NAME)

clean:
	rm -rf $(NAME)

re: clean all

.PHONY: all clean re
