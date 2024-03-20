#ifndef MYEXCEPTION_H
#define MYEXCEPTION_H

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

class my_out_of_range : public std::out_of_range
{
public:
    my_out_of_range(const std::string &arg, const char *file, int line) :
    std::out_of_range(arg) {
        std::ostringstream o;
        o << file << ":" << line << ": my_out_of_range: " << arg;
        msg = o.str();
    }
    ~my_out_of_range() throw() {}
    const char *what() const throw() {
        return msg.c_str();
    }

private:
    std::string msg;
};
#define throw_my_out_of_range(arg) throw my_out_of_range(arg, __FILE__, __LINE__);

std::string getOutOfRangeMsg(const int index, const int detect_line_count); 

#endif
