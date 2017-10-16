package algorithm;

/**
 * Created by MySsir on 2017/3/13.
 */
public class DesignModel {

}

//重写hashCode() equals()
class Coder {
    private String name;
    private int age;

    public int hashCode() {
        int result = 17;
        result = result * 31 + name.hashCode();
        result = result * 31 + age;
        return result;
    }

    public boolean equals(Object object) {
        //1.先判断是否等于自身
        if (object == this)
            return true;
        //2.使用instanceof运算符判断 other 是否为Coder类型的对象
        if (!(object instanceof Coder))
            return false;
        //3.比较Coder类中你自定义的数据域，name和age，一个都不能少
        Coder coder = (Coder)object;
        return (coder.age == age) && (coder.name.equals(name));
    }
}

/*
    工厂设计模式
    程序在接口和子类之间加入一个过渡端，通过此过渡端可以动态取得实现了共同接口的子类实例化对象
 */
//简单工厂模式例子
interface Animal {
    public void say();
}

class Cat implements Animal {
    @Override
    public void say() {
        System.out.println("I'm Cat");
    }
}

class Dog implements Animal {
    @Override
    public void say() {
        System.out.println("I'm Dog");
    }
}

class Factory {
    public static Animal getInstance(String className) {
        Animal a = null;
        if ("Cat".equals(className)) {
            a = new Cat();
        }
        if ("Dog".equals(className)) {
            a = new Dog();
        }
        return a;
    }
}

/*
    单例模式：
    所谓单例模式简单来说就是无论程序如何运行，采用单例模式设计的类（Singleton类）永远只会有一个实例化对象产生。具体的实现步骤如下：
    （1）、将采用单例设计模式的类的构造方法私有化（采用private修饰）。
    （2）、在其内部产生该类的实例化对象，并将其封装成private static类型。
    懒汉式单例模式
    第一次使用单例的时候才实例化对象，避免了在在类创建时候就实例化对象，但是没有及时使用造成的内存资源的浪费；
    仅在第一次调用单例的才会进行同步，同样线程安全，同时也避免了每次都同步造成的性能消耗
 */
class Singleton {
    private Singleton() {}
    private static Singleton singleton = null;
    //静态工厂方法
    public static  Singleton getInstance() {
        if (singleton == null) {
            synchronized (Singleton.class) {
                if (singleton == null) {
                    singleton = new Singleton();
                }
            }
        }
        return singleton;
    }
}

/*
    饿汉式单例模式
 */
class Singleton_1 {
    private Singleton_1() {}
    private static final Singleton_1 singleton_1 = new Singleton_1();
    //静态工厂方法
    public static Singleton_1 getInstance() {
        return singleton_1;
    }
}
