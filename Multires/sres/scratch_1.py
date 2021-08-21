# how to refer to data??????
# weights as properties of child
# how to traverse?
import numpy as np


# Python program to create a Complete Binary Tree from
# its linked list representation

# Linked List node
# Python program to create a Complete Binary Tree from
# its linked list representation

# Linked List node
class ListNode:

		# Constructor to create a new node
		def __init__(self, data):
			self.data = data
			self.next = None

# Binary Tree Node structure
class BinaryTreeNode:

	# Constructor to create a new node
	def __init__(self, data):
		self.data = data
		self.left = None
		self.right = None

# Class to convert the linked list to Binary Tree
class Conversion:

	# Constructor for storing head of linked list
	# and root for the Binary Tree
	def __init__(self, data = None):
		self.head = None
		self.root = None

	def push(self, new_data):

		# Creating a new linked list node and storing data
		new_node = ListNode(new_data)

		# Make next of new node as head
		new_node.next = self.head

		# Move the head to point to new node
		self.head = new_node

	def convertList2Binary(self):

		# Queue to store the parent nodes
		q = []

		# Base Case
		if self.head is None:
			self.root = None
			return

		# 1.) The first node is always the root node,
		# and add it to the queue
		self.root = BinaryTreeNode(self.head.data)
		q.append(self.root)

		# Advance the pointer to the next node
		self.head = self.head.next

		# Until th end of linked list is reached, do:
		while(self.head):

			# 2.a) Take the parent node from the q and
			# and remove it from q
			parent = q.pop(0) # Front of queue

			# 2.c) Take next two nodes from the linked list.
			# We will add them as children of the current
			# parent node in step 2.b.
			# Push them into the queue so that they will be
			# parent to the future node
			leftChild= None
			rightChild = None

			leftChild = BinaryTreeNode(self.head.data)
			q.append(leftChild)
			self.head = self.head.next
			if(self.head):
				rightChild = BinaryTreeNode(self.head.data)
				q.append(rightChild)
				self.head = self.head.next

			#2.b) Assign the left and right children of parent
			parent.left = leftChild
			parent.right = rightChild

	def inorderTraversal(self, root):
		if(root):
			self.inorderTraversal(root.left)
			print(root.data)
			self.inorderTraversal(root.right)

# Driver Program to test above function

# Object of conversion class
conv = Conversion()
conv.push(36)
conv.push(30)
conv.push(25)
conv.push(15)
conv.push(12)
conv.push(10)

conv.convertList2Binary()

print("Inorder Traversal of the contructed Binary Tree is:")
conv.inorderTraversal(conv.root)

# This code is contributed by Nikhil Kumar Singh(nickzuck_007)



# class Node():
#     def __init__(self, weight, parent=None, child1=None, child2=None):
#         self.weight = weight
#         self.parent = parent
#         self.child1 = child1
#         self.child2 = child2
#         # self.res1 = None
#         # self.res1 = None
#         self.children = []
#
#     def add_children(self, child):
#         self.children.append(child)
#
#     # def weight(self):
#     #     pass
#
#     def children_weight(self):
#         for c in self.children:
#             c.weight = 0 # calculate weight
#
#     def update_weight(self, weight):
#         self.weight = weight
#
#     def choose_child(self):
#         choice_weight = [c.weight for c in self.children]
#         return self.children[np.argmax(choice_weight)]
#
#
# for l in range(3):



# need to add children

# class Tree(Node):
#     def __init__(self, res=2, layer=3):
#         super().__init__()
#         for r in range(res):
#             for l in range(layer):

# a = Node(0)
# b = Node(0)
# c = Node(0)
# d = Node(0)
# e = Node(0)
# f = Node(0)
# g = Node(0)
# h = Node(0)
#
# a.add_children(b)
# a.add_children(c)
#
# b.add_children(d)
# b.add_children(e)
# b.add_children(f)
#
# f.add_children(g)
# f.add_children(h)
#
#
#
#
# print([child.weight for child in a.children])

