# Node class represents each element (node) in the list
class Node:
    def __init__(self, data):
        self.data = data  # Data of the node
        self.next = None  # Pointer to the next node

# LinkedList class will manage the list
class LinkedList:
    def __init__(self):
        self.head = None  # Initialize the head of the list

    # Method to add a node at the end of the list
    def append(self, data):
        new_node = Node(data)
        # If the list is empty, the new node is the head
        if self.head is None:
            self.head = new_node
            return
        # Traverse to the end of the list and add the new node
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node

    # Method to display the list
    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")  # End of the list

    # Method to insert at the beginning
    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    # Method to delete a node by its value
    def delete(self, key):
        current = self.head

        # If the head node itself holds the key
        if current and current.data == key:
            self.head = current.next
            current = None
            return

        # Search for the key to be deleted
        prev = None
        while current and current.data != key:
            prev = current
            current = current.next

        # If the key was not present
        if current is None:
            return

        # Unlink the node from the list
        prev.next = current.next
        current = None

# Usage Example
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.insert_at_beginning(0)
ll.display()  # Output: 0 -> 1 -> 2 -> 3 -> None

ll.delete(2)
ll.display()  # Output: 0 -> 1 -> 3 -> None
