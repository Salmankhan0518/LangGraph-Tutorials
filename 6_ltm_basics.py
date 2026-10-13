from langgraph.store.memory import InMemoryStore

# creating a store
store = InMemoryStore()

# creating a namespace
namespace = ("user", "u1")

# creating memory
# adding memory
store.put(namespace, "1", {"data": "User likes pizza"})
store.put(namespace, "2", {"data": "User preffers drink mode"})


# another namespace
namespace2 = ("user", "u2")

# adding memory
store.put(namespace2, "1", {"data": "User likes pasta"})
store.put(namespace2, "2", {"data": "User likes grid style navigation"})

# Retrieving Memories
# store.get(namespace, key)
result = store.get(namespace, "2") 
print(result)


# Retrieving all Memories
items = store.search(namespace)

for item in items:
    print(item.value)
