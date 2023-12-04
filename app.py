from Enron_dataset.file_reader import File_reader

fr = File_reader()

data, label = fr.load_ham_and_spam(ham_paths = "default", spam_paths = "default", max = 3000)

print(data)
print(label)
