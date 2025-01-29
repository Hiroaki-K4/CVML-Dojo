SRCS := main.cu
INCLUDE := -I.
NAME := stream
CXX := nvcc

all: $(NAME)

$(NAME): $(SRCS)
	$(CXX) $(SRCS) $(INCLUDE) -o $(NAME)

clean:
	rm -rf $(NAME)

re: clean all

.PHONY: all clean re
