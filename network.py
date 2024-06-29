from abc import ABC, abstractmethod

class Layer(ABC):

    @abstractmethod
    def forward_pass(self):
        pass

    @abstractmethod
    def backward_pass(self):
        pass

class Fuzzification(Layer):

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

class Rule(Layer):
        
    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

class Normalization(Layer):

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

class Defuzzification(Layer):

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

class Output(Layer):

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass