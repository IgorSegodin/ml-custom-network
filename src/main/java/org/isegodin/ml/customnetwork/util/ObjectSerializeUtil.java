package org.isegodin.ml.customnetwork.util;

import lombok.SneakyThrows;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Base64;

/**
 * @author isegodin
 */
public class ObjectSerializeUtil {

    @SneakyThrows
    public static void main(String[] args) {
        Runnable r = (Runnable & Serializable)() -> System.out.println("Serializable!");

        ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
        ObjectOutputStream outputStream = new ObjectOutputStream(byteOut);
        outputStream.writeObject(r);

        String bytesAsText = Base64.getEncoder().encodeToString(byteOut.toByteArray());

        ObjectInputStream inputStream = new ObjectInputStream(new ByteArrayInputStream(Base64.getDecoder().decode(bytesAsText)));

        Runnable read = (Runnable) inputStream.readObject();

        read.run();

        System.out.println(bytesAsText);
    }
}
