#include <pybind11/pybind11.h>
#include <exiv2/exiv2.hpp>
#include <string>
#include <sstream>
#include <iostream>

namespace py = pybind11;
std::stringstream error_log;

class Buffer{
public:
    char *data;
    long size;

    Buffer(const char *data_, long size_){
        size = size_;
        data = (char *)calloc(size, sizeof(char));
        if(data == NULL)
            throw std::runtime_error("Failed to allocate memory.");
        memcpy(data, data_, size);
    }

    void destroy(){
        if(data){
            // std::cout << "deleting" << data << std::endl;
            free(data);
            data = NULL;
        }
    }

    py::bytes dump(){
        return py::bytes((char *)data, size);
    }
};

void check_error_log()
{
    std::string str = error_log.str();
    if(str != ""){
        error_log.clear();  // Clear it so it can be used again
        error_log.str("");
        throw std::runtime_error(str);
    }
}

void logHandler(int level, const char *msg)
{
    switch (level)
    {
    case Exiv2::LogMsg::debug:
    case Exiv2::LogMsg::info:
    case Exiv2::LogMsg::warn:
        std::cout << msg << std::endl;
        break;

    case Exiv2::LogMsg::error:
        // For unknown reasons, the exception thrown here cannot be caught, so save the log to error_log
        // throw std::exception(msg);
        error_log << msg;
        break;

    default:
        return;
    }
}

void set_log_level(int level)
{
    if (level == 0)
        Exiv2::LogMsg::setLevel(Exiv2::LogMsg::debug);
    if (level == 1)
        Exiv2::LogMsg::setLevel(Exiv2::LogMsg::info);
    if (level == 2)
        Exiv2::LogMsg::setLevel(Exiv2::LogMsg::warn);
    if (level == 3)
        Exiv2::LogMsg::setLevel(Exiv2::LogMsg::error);
    if (level == 4)
        Exiv2::LogMsg::setLevel(Exiv2::LogMsg::mute);
}

void init()
{
    Exiv2::LogMsg::setHandler(logHandler);
}

#define read_block                                                     \
    {                                                                  \
        py::list table;                                                \
        for (; i != end; ++i)                                          \
        {                                                              \
            py::list line;                                             \
            line.append(py::bytes(i->key()));                          \
                                                                       \
            std::stringstream _value;                                  \
            _value << i->value();                                      \
            line.append(py::bytes(_value.str()));                      \
                                                                       \
            const char *typeName = i->typeName();                      \
            line.append(py::bytes((typeName ? typeName : "Unknown"))); \
            table.append(line);                                        \
        }                                                              \
        check_error_log();                                             \
        return table;                                                  \
    }

class Image{
public:
    Exiv2::Image::AutoPtr *img = new Exiv2::Image::AutoPtr;

    Image(const char *filename){
        *img = Exiv2::ImageFactory::open(filename);
        if (img->get() == 0)
            throw Exiv2::Error(Exiv2::kerErrorMessage, "Can not open this image.");
        (*img)->readMetadata();     // Calling readMetadata() reads all types of metadata supported by the image
        check_error_log();
    }

    Image(Buffer buffer){
        *img = Exiv2::ImageFactory::open((Exiv2::byte *)buffer.data, buffer.size);
        if (img->get() == 0)
            throw Exiv2::Error(Exiv2::kerErrorMessage, "Can not open this image.");
        (*img)->readMetadata();
        check_error_log();
    }

    void close_image()
    {
        delete img;
        check_error_log();
    }

    py::bytes get_bytes_of_image()
    {
        Exiv2::BasicIo &io = (*img)->io();
        return py::bytes((char *)io.mmap(), io.size());
    }

    py::object read_comment()
    {
        return py::bytes((*img)->comment());
    }

    void modify_comment(py::str data, py::str encoding)
    {
        std::string data_str = py::bytes(data.attr("encode")(encoding));
        (*img)->setComment(data_str);
        (*img)->writeMetadata();
        check_error_log();
    }

    void clear_comment()
    {
        (*img)->clearComment();
        (*img)->writeMetadata();
        check_error_log();
    }

};

PYBIND11_MODULE(exif, m)
{
    m.doc() = "Expose the API of exiv2 to Python.";
    m.def("set_log_level", &set_log_level);
    m.def("init"         , &init);
    py::class_<Buffer>(m, "Buffer")
        .def(py::init<const char *, long>())
        .def_readonly("data"      , &Buffer::data)
        .def_readonly("size"      , &Buffer::size)
        .def("destroy"            , &Buffer::destroy)
        .def("dump"               , &Buffer::dump);
    py::class_<Image>(m, "Image")
        .def(py::init<const char *>())
        .def(py::init<Buffer &>())
        .def("close_image"       , &Image::close_image)
        .def("get_bytes_of_image", &Image::get_bytes_of_image)
        .def("read_comment"      , &Image::read_comment)
        .def("modify_comment"    , &Image::modify_comment)
        .def("clear_comment"     , &Image::clear_comment);
}
