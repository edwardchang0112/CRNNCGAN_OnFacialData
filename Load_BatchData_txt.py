def Read_batch_files_fromtxt(txtfile_name):
    # open file
    file_list = []
    with open(txtfile_name, 'r') as fp:
        all_lines = fp.readlines()
        for line in all_lines:
            line = line.strip('\n')
            line = line.strip()
            file_list.append(line)
    return file_list

if __name__ == '__main__':
    txtfile_name = 'All_User_Files.txt'
    file_list = Read_batch_files_fromtxt(txtfile_name)
    print("file_list = ", file_list)
