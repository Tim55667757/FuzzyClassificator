__author__ = 'Tom Schaul, tom@idsia.ch'

from inspect import isclass

from xml.dom.minidom import parse, getDOMImplementation
from pybrain.utilities import fListToString
from scipy import zeros
import string


class XMLHandling:
    """ general purpose methods for reading, writing and editing XML files.
    This class should wrap all the XML-specific code, and then be subclassed
    by specialized readers/writers that use its methods.

    The priority is on readability and usability for the subclasses, not efficiency.
    """

    def __init__(self, filename, newfile):
        """ :key newfile: is the file to be read or is it a new file? """
        self.filename = filename
        if not newfile:
            self.dom = parse(filename)
            if self.dom.firstChild.nodeName != 'PyBrain':
                raise Exception('Not a correct PyBrain XML file')
        else:
            domimpl = getDOMImplementation()
            self.dom = domimpl.createDocument(None, 'PyBrain', None)
        self.root = self.dom.documentElement

    def save(self):
        file = open(self.filename, 'w')
        file.write(self.dom.toprettyxml())
        file.close()

    def readAttrDict(self, node, transform = None):
        """ read a dictionnary of attributes
        :key transform: optionally function transforming the attribute values on reading """
        args = {}
        for name, val in node.attributes.items():
            name = str(name)
            if transform != None:
                args[name] = transform(val, name)
            else:
                args[name] = val
        return args

    def writeAttrDict(self, node, adict, transform = None):
        """ read a dictionnary of attributes

        :key transform: optionally transform the attribute values on writing """
        for name, val in adict.items():
            if val != None:
                if transform != None:
                    node.setAttribute(name, transform(val, name))
                else:
                    node.setAttribute(name, val)

    def newRootNode(self, name):
        return self.newChild(self.root, name)

    def newChild(self, node, name):
        """ create a new child of node with the provided name. """
        elem = self.dom.createElement(name)
        node.appendChild(elem)
        return elem

    def addTextNode(self, node, text):
        tmp = self.dom.createTextNode(text)
        node.appendChild(tmp)

    def getChild(self, node, name):
        """ get the child with the given name """
        for n in node.childNodes:
            if name and n.nodeName == name:
                return n

    def getChildrenOf(self, node):
        """ get the element children """
        return filter(lambda x: x.nodeType == x.ELEMENT_NODE, node.childNodes)

    def findNode(self, name, index = 0, root = None):
        """ return the toplevel node with the provided name (if there are more, choose the
        index corresponding one). """
        if root == None:
            root = self.root
        for n in root.childNodes:
            if n.nodeName == name:
                if index == 0:
                    return n
                index -= 1
        return None

    def findNamedNode(self, name, nameattr, root = None):
        """ return the toplevel node with the provided name, and the fitting 'name' attribute. """
        if root == None:
            root = self.root
        for n in root.childNodes:
            if n.nodeName == name:
            # modif JPQ
            #                if 'name' in n.attributes:
                if n.attributes['name']:
                # modif JPQ
                #                    if n.attributes['name'] == nameattr:
                    if n.attributes['name'].value == nameattr:
                        return n
        return None

    def writeDoubles(self, node, l, precision = 6):
        self.addTextNode(node, fListToString(l, precision)[2:-1])

    def writeMatrix(self, node, m, precision = 6):
        for i, row in enumerate(m):
            r = self.newChild(node, 'row')
            self.writeAttrDict(r, {'number':str(i)})
            self.writeDoubles(r, row, precision)

    def readDoubles(self, node):
        dstrings = string.split(node.firstChild.data)
        return map(lambda s: float(s), dstrings)

    def readMatrix(self, node):
        rows = []
        for c in self.getChildrenOf(node):
            rows.append(self.readDoubles(c))
        if len(rows) == 0:
            return None
        res = zeros((len(rows), len(rows[0])))
        for i, r in enumerate(rows):
            res[i] = r
        return res


def baseTransform(val):
    """ back-conversion: modules are encoded by their name
    and classes by the classname """
    from pybrain.structure.modules.module import Module
    from inspect import isclass

    if isinstance(val, Module):
        return val.name
    elif isclass(val):
        return val.__name__
    else:
        return str(val)


from pybrain.structure.connections.shared import SharedConnection
from pybrain.structure.networks.network import Network
from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain.utilities import canonicClassString

# TODO: higher precision on writing parameters


class NetworkWriter(XMLHandling):
    """ A class that can take a network and write it to an XML file """

    @staticmethod
    def appendToFile(net, filename):
        """ append the network to an existing xml file """
        w = NetworkWriter(filename, newfile = False)
        netroot = w.newRootNode('Network')
        w.writeNetwork(net, netroot)
        w.save()

    @staticmethod
    def writeToFile(net, filename):
        """ write the network as a new xml file """
        w = NetworkWriter(filename, newfile = True)
        netroot = w.newRootNode('Network')
        w.writeNetwork(net, netroot)
        w.save()

    def writeNetwork(self, net, netroot):
        """ write a Network into a new XML node """
        netroot.setAttribute('name', net.name)
        netroot.setAttribute('class', canonicClassString(net))
        if net.argdict:
            self.writeArgs(netroot, net.argdict)

        # the modules
        mods = self.newChild(netroot, 'Modules')
        # first write the input modules (in order)
        for im in net.inmodules:
            self.writeModule(mods, im, True, im in net.outmodules)
        # now the output modules (in order)
        for om in net.outmodules:
            if om not in net.inmodules:
                self.writeModule(mods, om, False, True)
        # now the rest
        for m in net.modulesSorted:
            if m not in net.inmodules and m not in net.outmodules:
                self.writeModule(mods, m, False, False)

        # the motherconnections
        if len(net.motherconnections) > 0:
            mothers = self.newChild(netroot, 'MotherConnections')
            for m in net.motherconnections:
                self.writeBuildable(mothers, m)

        # the connections
        conns = self.newChild(netroot, 'Connections')
        for m in net.modulesSorted:
            for c in net.connections[m]:
                self.writeConnection(conns, c, False)
        if hasattr(net, "recurrentConns"):
            for c in net.recurrentConns:
                self.writeConnection(conns, c, True)

    def writeModule(self, rootnode, m, inmodule, outmodule):
        if isinstance(m, Network):
            mnode = self.newChild(rootnode, 'Network')
            self.writeNetwork(m, mnode)
        else:
            mnode = self.writeBuildable(rootnode, m)
        if inmodule:
            mnode.setAttribute('inmodule', 'True')
        elif outmodule:
            mnode.setAttribute('outmodule', 'True')

    def writeConnection(self, rootnode, c, recurrent):
        mnode = self.writeBuildable(rootnode, c)
        if recurrent:
            mnode.setAttribute('recurrent', 'True')

    def writeBuildable(self, rootnode, m):
        """ store the class (with path) and name in a new child. """
        mname = m.__class__.__name__
        mnode = self.newChild(rootnode, mname)
        mnode.setAttribute('name', m.name)
        mnode.setAttribute('class', canonicClassString(m))
        if m.argdict:
            self.writeArgs(mnode, m.argdict)
        if m.paramdim > 0 and not isinstance(m, SharedConnection):
            self.writeParams(mnode, m.params)
        return mnode

    def writeArgs(self, node, argdict):
        """ write a dictionnary of arguments """
        for name, val in argdict.items():
            if val != None:
                tmp = self.newChild(node, name)
                if isclass(val):
                    s = canonicClassString(val)
                else:
                    s = getattr(val, 'name', repr(val))
                tmp.setAttribute('val', s)

    def writeParams(self, node, params):
        # TODO: might be insufficient precision
        pnode = self.newChild(node, 'Parameters')
        self.addTextNode(pnode, str(list(params)))
